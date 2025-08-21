import streamlit as st
import pandas as pd
from PIL import Image
import io
from google.cloud import storage
from google.cloud import bigquery
import os
from langdetect import detect
import json
import re
from datetime import datetime, timedelta
from googletrans import Translator
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from logger import logger

# Initialize BLIP Large model
@st.cache_resource
def init_blip_large():
    """Initialize BLIP Large model for image captioning"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        model.to(device)
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading BLIP Large model: {str(e)}")
        return None, None, None

# Function to generate BLIP Large caption
def generate_blip_caption(image, processor=None, model=None, device=None):
    """Generate caption using BLIP Large model"""
    try:
        if processor is None or model is None:
            return "BLIP Large model not available"
            
        # Process image
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# Initialize translator
@st.cache_resource
def init_translator():
    return Translator()

# Initialize BigQuery client
@st.cache_resource
def init_bigquery_client():
    return bigquery.Client(project='swift-catfish-337215')

# Initialize GCS client
@st.cache_resource
def init_gcs_client():
    return storage.Client(project='scope3-prod')
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_cached(date, limit, classification_values):
    """
    UPDATED Query for Instagram and Facebook video content only using new table structure
    """
    try:
        client = init_bigquery_client()
        
        if isinstance(classification_values, list):
            classification_conditions = " OR ".join([f"ac.common_sense_classification_value = {val}" for val in classification_values])
            where_clause = f"({classification_conditions})"
        else:
            where_clause = f"ac.common_sense_classification_value = {classification_values}"
        
        # Updated query to use new table and schema
        query = f"""
        WITH unique_content AS (
          SELECT
            ac.artifact_id,
            ac.common_sense_classification_value,
            ac.classified_at,
            ac.categories,
            ac.reasoning,
            ac.artifact_title,
            ac.artifact_lang,
            ac.artifact_url,
            ac.source,
            ac.artifact_bucket,
            ac.artifact_filepath,
            -- Note: Using artifact_url from new table
            ROW_NUMBER() OVER (
              PARTITION BY ac.artifact_id
              ORDER BY ac.classified_at DESC
            ) AS artifact_rn,
            ROW_NUMBER() OVER (
              PARTITION BY ac.artifact_url
              ORDER BY ac.classified_at DESC
            ) AS url_rn
          FROM `swift-catfish-337215.artifacts.artifact_classification` ac
          WHERE DATE(ac.classified_at) = "{date}"
          AND {where_clause}
          AND (
            ac.artifact_url LIKE '%instagram.com%' 
            OR ac.artifact_url LIKE '%facebook.com%'
            OR ac.artifact_url LIKE '%fb.com%'
          )
        )
        
        SELECT
          artifact_id,
          common_sense_classification_value,
          classified_at,
          categories,
          reasoning,
          artifact_title,
          artifact_lang,
          artifact_url,
          source,
          artifact_bucket,
          artifact_filepath
        FROM unique_content
        WHERE artifact_rn = 1  -- Latest classification for each artifact_id
        AND url_rn = 1         -- Latest record for each URL
        ORDER BY classified_at DESC
        LIMIT {limit * 5}      -- Get more records since we're filtering for videos
        """
        
        df = client.query(query).to_dataframe()
        
        # Filter for video content on client side (much faster)
        if len(df) > 0:
            video_mask = (
                df['artifact_url'].str.contains('/reel/|/video|/videos/', case=False, na=False, regex=True) |
                df['artifact_title'].str.contains('video|reel', case=False, na=False)
            )
            df = df[video_mask].head(limit)
        
        logger.debug({"rows": len(df), "date": date, "classification_values": classification_values}, 'loaded unique Instagram/Facebook video data from BigQuery')
        return df
    except Exception as e:
        logger.error({"error": str(e)}, 'failed to load unique Instagram/Facebook video data from BigQuery')
        raise e

# Function to get image path based on bucket structure
def get_image_path(bucket_name, json_path, image_identifier):
    try:
        # If the image identifier is a full GCS path, extract just the path part
        if image_identifier.startswith('gs://'):
            match = re.match(r'gs://[^/]+/(.+)', image_identifier)
            if match:
                return match.group(1)
        
        # If the image identifier already has the full path, use it as is
        if image_identifier.startswith('artifacts/assets/'):
            return image_identifier
            
        # Otherwise, construct the path
        return f"artifacts/assets/{image_identifier}"
    except Exception as e:
        st.error(f"Error constructing image path: {str(e)}")
        return None

# Function to read JSON from GCS
def read_json_from_gcs(bucket_name, json_path):
    try:
        client = init_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(json_path)

        # Check if blob exists
        if not blob.exists():
            st.error(f"JSON file not found: gs://{bucket_name}/{json_path}")
            return None

        content = blob.download_as_string()
        json_data = json.loads(content.decode('utf-8'))
        return json_data
    except Exception as e:
        st.error(f"Error loading JSON gs://{bucket_name}/{json_path}: {str(e)}")
        return None

# Function to load image from GCS
def load_image_from_gcs(bucket_name, image_path):
    try:
        client = init_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(image_path)

        # Check if blob exists
        if not blob.exists():
            st.error(f"Image file not found: gs://{bucket_name}/{image_path}")
            return None

        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        st.error(f"Error loading image gs://{bucket_name}/{image_path}: {str(e)}")
        return None

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# UPDATED Function to query BigQuery by Artifact ID - Instagram/Facebook only
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_by_artifact_id(artifact_id, classification_value=None):
    """
    UPDATED Query BigQuery for a specific Artifact ID using new table structure
    """
    try:
        client = init_bigquery_client()
        
        where_conditions = [f'ac.artifact_id = "{artifact_id}"']
        if classification_value is not None:
            where_conditions.append(f'ac.common_sense_classification_value = {classification_value}')
        
        where_clause = " AND ".join(where_conditions)
        
        # Updated query to use new table structure
        query = f"""
        SELECT
          ac.artifact_id,
          ac.common_sense_classification_value,
          ac.classified_at,
          ac.categories,
          ac.reasoning,
          ac.artifact_title,
          ac.artifact_lang,
          ac.artifact_url,
          ac.source,
          ac.model_id,
          ac.ai_model_name,
          ac.classification_mode,
          ac.artifact_bucket,
          ac.artifact_filepath
        FROM `swift-catfish-337215.artifacts.artifact_classification` ac
        WHERE {where_clause}
        AND (
          ac.artifact_url LIKE '%instagram.com%'
          OR ac.artifact_url LIKE '%facebook.com%'
          OR ac.artifact_url LIKE '%fb.com%'
        )
        ORDER BY ac.classified_at DESC
        LIMIT 1
        """
        
        df = client.query(query).to_dataframe()
        logger.debug({"rows": len(df), "artifact_id": artifact_id, "classification_value": classification_value}, 'loaded Instagram/Facebook data by Artifact ID')
        return df
    except Exception as e:
        logger.error({"error": str(e)}, 'failed to load Instagram/Facebook data by Artifact ID')
        raise e

# Function to load artifact content from GCS
def load_artifact_content(artifact_bucket, artifact_filepath):
    """Load artifact content (frames/assets) from GCS based on bucket and filepath"""
    try:
        if not artifact_bucket or not artifact_filepath:
            return None, []
            
        client = init_gcs_client()
        bucket = client.bucket(artifact_bucket)
        blob = bucket.blob(artifact_filepath)
        
        if not blob.exists():
            return None, []
            
        # Download and parse the artifact JSON
        content = blob.download_as_string()
        artifact_data = json.loads(content.decode('utf-8'))
        
        # Extract frame information
        frames = []
        if 'text_content' in artifact_data:
            text_content = artifact_data['text_content']
            
            if isinstance(text_content, list):
                for item in text_content:
                    if isinstance(item, dict):
                        item_type = item.get('type', 'unknown')
                        
                        # Handle video frames
                        if item_type == 'video' and 'video' in item:
                            video_data = item['video']
                            if 'frames' in video_data and isinstance(video_data['frames'], list):
                                for frame in video_data['frames']:
                                    if isinstance(frame, dict) and 'filepath' in frame:
                                        frames.append(frame)
                        
                        # Handle direct video items (fallback)
                        elif item_type == 'video' and 'frames' in item:
                            for frame in item['frames']:
                                if isinstance(frame, dict) and 'filepath' in frame:
                                    frames.append(frame)
                        
                        # Handle image assets
                        elif item_type == 'image' and 'image' in item:
                            image_data = item['image']
                            if 'filepath' in image_data:
                                frames.append({
                                    'filepath': image_data['filepath'],
                                    'caption': image_data.get('caption', ''),
                                    'alt_text': image_data.get('alt_text', ''),
                                    'type': 'image'
                                })
        
        return artifact_data, frames
        
    except Exception as e:
        return None, []

# Function to display the main interface content
def display_main_interface(page_type):
    """Display the main interface for either CSBS or Safe videos"""
    
    # Page-specific session state
    page_key = f"{page_type.lower()}_"
    
    # Initialize page-specific session state
    if f'{page_key}current_index' not in st.session_state:
        st.session_state[f'{page_key}current_index'] = 0
    if f'{page_key}data' not in st.session_state:
        st.session_state[f'{page_key}data'] = None
    if f'{page_key}current_frame_index' not in st.session_state:
        st.session_state[f'{page_key}current_frame_index'] = 0
    if f'{page_key}current_artifact_data' not in st.session_state:
        st.session_state[f'{page_key}current_artifact_data'] = None
    if f'{page_key}current_frames' not in st.session_state:
        st.session_state[f'{page_key}current_frames'] = []

    # Use page-specific state and update local variables
    current_index = st.session_state[f'{page_key}current_index']
    data = st.session_state[f'{page_key}data']
    current_frame_index = st.session_state[f'{page_key}current_frame_index']
    current_artifact_data = st.session_state[f'{page_key}current_artifact_data']
    current_frames = st.session_state[f'{page_key}current_frames']

    # Page title
    if page_type == "CSBS":
        st.title("🚨 Meta Video CSBS Review")
        st.markdown("*Review Meta video content classified as below common sense floor*")
    else:
        st.title("✅ Meta Video Safe Review")
        st.markdown("*Review Meta video content classified as safe*")

    # Filter controls
    st.markdown("### Data Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Date picker
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now().date() - timedelta(days=1),
            max_value=datetime.now().date(),
            key=f"{page_key}date"
        )

    with col2:
        # Limit selector with custom input option
        limit_type = st.radio(
            "Sample Size Type:",
            ["Quick Select", "Custom"],
            horizontal=True,
            key=f"{page_key}limit_type"
        )
        
        if limit_type == "Quick Select":
            limit_options = [1, 5, 10, 25, 50, 100, 200, 500]
            selected_limit = st.selectbox("Number of Records", limit_options, index=2, key=f"{page_key}limit_quick")
        else:
            selected_limit = st.number_input(
                "Custom Sample Size", 
                min_value=1, 
                max_value=10000, 
                value=10, 
                step=1,
                key=f"{page_key}limit_custom"
            )
            if selected_limit > 1000:
                st.warning("⚠️ Large sample sizes may take longer to load")
            elif selected_limit > 100:
                st.info("ℹ️ Sample size > 100 may take a few extra seconds")

    with col3:
        # Classification value selector - NOW AVAILABLE ON BOTH PAGES
        classification_options = [0, 100, "Any"]
        # Set default based on page type but allow user to change
        if page_type == "CSBS":
            default_classification = 0
        else:
            default_classification = 100
        
        # Find the index for the default value
        default_index = classification_options.index(default_classification)
        
        selected_classification = st.selectbox(
            "Classification Value", 
            classification_options, 
            index=default_index,
            key=f"{page_key}classification_filter"
        )

    with col4:
        # Cache management and BLIP settings
        blip_col1, blip_col2 = st.columns(2)
        
        with blip_col1:
            if st.button("Clear Cache", key=f"{page_key}clear_cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with blip_col2:
            use_blip_large = st.checkbox(
                "Use BLIP Large",
                value=True,
                help="Generate new captions with BLIP Large model",
                key=f"{page_key}use_blip"
            )

    # Search section
    st.markdown("### Search Options")
    search_col1, search_col2, search_col3 = st.columns([2, 1, 1])

    with search_col1:
        artifact_id_input = st.text_input("Artifact ID (optional)", placeholder="e.g., meta:aWdfbWVkaWFfM3B2OjE4Mjc4MTMzMDU1MjYwMjU4", key=f"{page_key}artifact_id")

    with search_col2:
        # Auto-adjust limit based on search type
        if artifact_id_input.strip():
            st.info("Limit set to 1 for Artifact ID search")
            effective_limit = 1
        else:
            effective_limit = selected_limit

    with search_col3:
        if st.button("Load Data", key=f"{page_key}load_data"):
            if artifact_id_input.strip():
                # Search by Artifact ID
                try:
                    with st.spinner("Searching by Artifact ID..."):
                        search_classification = selected_classification if selected_classification != "Any" else None
                        loaded_data = query_bigquery_by_artifact_id(artifact_id_input.strip(), search_classification)
                        if len(loaded_data) > 0:
                            st.session_state[f'{page_key}data'] = loaded_data
                            st.session_state[f'{page_key}current_index'] = 0
                            st.session_state[f'{page_key}current_frame_index'] = 0
                            data = loaded_data
                            current_index = 0
                            current_frame_index = 0
                            st.success(f"Found {len(loaded_data)} Meta video record(s)")
                        else:
                            st.warning(f"No Meta video records found for Artifact ID: {artifact_id_input}")
                except Exception as e:
                    st.error(f"Error searching by Artifact ID: {str(e)}")
            else:
                # Search by Date
                try:
                    with st.spinner("Loading Instagram/Facebook video data..."):
                        if selected_classification == "Any":
                            # Use cached query for both classification values
                            loaded_data = query_bigquery_cached(selected_date.strftime('%Y-%m-%d'), effective_limit, [0, 100])
                            # Sort and limit the combined results
                            loaded_data = loaded_data.sort_values('classified_at', ascending=False).head(effective_limit)
                        else:
                            # Use cached query for single classification value
                            loaded_data = query_bigquery_cached(selected_date.strftime('%Y-%m-%d'), effective_limit, selected_classification)
                        
                        st.session_state[f'{page_key}data'] = loaded_data
                        st.session_state[f'{page_key}current_index'] = 0
                        st.session_state[f'{page_key}current_frame_index'] = 0
                        data = loaded_data
                        current_index = 0
                        current_frame_index = 0
                        st.success(f"✅ Loaded {len(loaded_data)} unique Meta video records")
                except Exception as e:
                    st.error(f"❌ Error loading video data: {str(e)}")

    # Check if data is loaded
    if data is None or len(data) == 0:
        st.info("Please click 'Load Data' to start reviewing Meta video content.")
        return
    
    # Get current item based on current index
    current_item = data.iloc[current_index].to_dict()

    # Load artifact content when record changes
    if f'{page_key}last_record_index' not in st.session_state or st.session_state[f'{page_key}last_record_index'] != current_index:
        try:
            artifact_bucket = current_item.get('artifact_bucket', '')
            artifact_filepath = current_item.get('artifact_filepath', '')
            artifact_data, frames = load_artifact_content(artifact_bucket, artifact_filepath)
            
            st.session_state[f'{page_key}current_artifact_data'] = artifact_data
            st.session_state[f'{page_key}current_frames'] = frames
            st.session_state[f'{page_key}current_frame_index'] = 0
            st.session_state[f'{page_key}last_record_index'] = current_index
            
            current_artifact_data = artifact_data
            current_frames = frames
            current_frame_index = 0
        except Exception as e:
            current_artifact_data = None
            current_frames = []

    # Navigation controls
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Previous", key=f"{page_key}prev"):
            if st.session_state[f'{page_key}current_index'] > 0:
                st.session_state[f'{page_key}current_index'] -= 1
                # Reset frame index when changing records
                st.session_state[f'{page_key}current_frame_index'] = 0
            st.rerun()

    with col2:
        if st.button("Next", key=f"{page_key}next"):
            if st.session_state[f'{page_key}current_index'] < len(data) - 1:
                st.session_state[f'{page_key}current_index'] += 1
                # Reset frame index when changing records
                st.session_state[f'{page_key}current_frame_index'] = 0
            st.rerun()

    with col3:
        st.write(f"Record {st.session_state[f'{page_key}current_index'] + 1} of {len(data)}")

    with col4:
        # Jump to specific record
        if len(data) > 0:
            jump_to = st.number_input("Jump to record", min_value=1, max_value=len(data), value=st.session_state[f'{page_key}current_index'] + 1, key=f"{page_key}jump")
            if jump_to != st.session_state[f'{page_key}current_index'] + 1:
                st.session_state[f'{page_key}current_index'] = jump_to - 1
                # Reset frame index when jumping to new record
                st.session_state[f'{page_key}current_frame_index'] = 0
                st.rerun()
        else:
            st.write("No records to navigate")

    # Display metadata with comprehensive debug info
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Metadata")
    
    # Debug: Show all available fields in the current item
    st.write("🔍 DEBUG: Available fields in current record:")
    available_fields = list(current_item.keys())
    st.write(f"   Fields: {available_fields}")
    
    # Updated to use new schema fields
    full_artifact_id = current_item.get('artifact_id', 'N/A')
    url = current_item.get('artifact_url', 'N/A')  # Updated field name
    classification = current_item.get('common_sense_classification_value', 'N/A')  # Updated field name
    
    st.write(f"🔍 DEBUG: Record Index: {current_index} of {len(data)}")
    st.write(f"🔍 DEBUG: Full Artifact ID: {full_artifact_id}")
    st.write(f"🔍 DEBUG: URL: {url}")
    st.write(f"🔍 DEBUG: Classification: {classification}")
    st.write(f"🔍 DEBUG: Artifact Bucket: {current_item.get('artifact_bucket', 'N/A')}")
    st.write(f"🔍 DEBUG: Artifact Filepath: {current_item.get('artifact_filepath', 'N/A')}")
    
    # Show if this is a duplicate by checking previous records
    if current_index > 0:
        prev_urls = []
        for i in range(min(current_index, 5)):  # Check last 5 records
            prev_item = data.iloc[i].to_dict()
            prev_urls.append(prev_item.get('artifact_url', ''))  # Updated field name
        
        if url in prev_urls:
            st.error(f"⚠️ DUPLICATE DETECTED: This URL appeared in previous records!")
        else:
            st.success(f"✅ UNIQUE: This URL is unique in the current dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Artifact ID:", full_artifact_id)
        st.write("URL:", url)
        st.write("Title:", current_item.get('artifact_title', 'N/A'))  # New field
    with col2:
        classification_value_display = current_item.get('common_sense_classification_value', 'N/A')
        if classification_value_display == 0:
            st.markdown('Classification Value: <span style="color: red; font-weight: bold; font-size: 18px;">🚨 Below Common Sense Floor</span>', unsafe_allow_html=True)
        elif classification_value_display == 100:
            st.markdown('Classification Value: <span style="color: green; font-weight: bold; font-size: 18px;">✅ Safe</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'Classification Value: <span style="font-weight: bold; font-size: 18px;">{classification_value_display}</span>', unsafe_allow_html=True)
        st.write("Classified At:", current_item.get('classified_at', 'N/A'))
        st.write("AI Model:", current_item.get('ai_model_name', 'N/A'))  # New field
    with col3:
        # Display content type - emphasize it's Meta video
        artifact_url = current_item.get('artifact_url', '')
        if 'instagram.com' in artifact_url:
            if '/reel/' in artifact_url.lower():
                content_type = "📱 Meta Reel"
            else:
                content_type = "📱 Meta Video"
        elif 'facebook.com' in artifact_url or 'fb.com' in artifact_url:
            if '/reel/' in artifact_url.lower():
                content_type = "📘 Meta Reel"
            else:
                content_type = "📘 Meta Video"
        else:
            content_type = "🎬 Video"
        
        st.markdown(f'**Content Type:** <span style="font-weight: bold; font-size: 16px; color: #1f77b4;">{content_type}</span>', unsafe_allow_html=True)
        st.write("Language:", current_item.get('artifact_lang', 'N/A'))
        st.write("Source:", current_item.get('source', 'N/A'))  # New field
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)

    # Main content area with two columns
    main_col1, main_col2 = st.columns([1, 1])

    with main_col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Frames")
        
        if current_frames and len(current_frames) > 0:
            # Frame navigation controls
            st.markdown("#### Frame Navigation")
            frame_nav_col1, frame_nav_col2, frame_nav_col3 = st.columns([1, 2, 1])
            
            with frame_nav_col1:
                if st.button("⬅️ Previous Frame", key=f"{page_key}prev_frame"):
                    if current_frame_index > 0:
                        st.session_state[f'{page_key}current_frame_index'] -= 1
                    st.rerun()
            
            with frame_nav_col2:
                st.write(f"Frame {current_frame_index + 1} of {len(current_frames)}")
            
            with frame_nav_col3:
                if st.button("Next Frame ➡️", key=f"{page_key}next_frame"):
                    if current_frame_index < len(current_frames) - 1:
                        st.session_state[f'{page_key}current_frame_index'] += 1
                    st.rerun()
            
            # Display current frame
            if current_frame_index < len(current_frames):
                current_frame = current_frames[current_frame_index]
                frame_path = current_frame.get('filepath', '')
                
                if frame_path:
                    # Get artifact bucket for image loading
                    artifact_bucket = current_item.get('artifact_bucket', '')
                    
                    if artifact_bucket:
                        try:
                            image = load_image_from_gcs(artifact_bucket, frame_path)
                            if image is not None:
                                st.image(image)
                                
                                # Generate BLIP Large caption
                                if use_blip_large:
                                    processor, model, device = init_blip_large()
                                    if processor is not None and model is not None:
                                        with st.spinner("Generating BLIP Large caption..."):
                                            blip_caption = generate_blip_caption(image, processor, model, device)
                                            st.markdown("**🤖 BLIP Large Caption:**")
                                            st.info(blip_caption)
                                    else:
                                        st.warning("BLIP Large model not available")
                                else:
                                    st.info("BLIP Large disabled - enable in filters above")
                            else:
                                st.error(f"Could not load frame: {frame_path}")
                        except Exception as e:
                            st.error(f"Error loading frame: {str(e)}")
                    else:
                        st.error("No artifact bucket specified for frame loading")
                        
                    # Display frame metadata
                    st.write(f"**Frame Path:** {frame_path}")
                    if 'timestamp' in current_frame:
                        st.write(f"**Timestamp:** {current_frame['timestamp']}ms")
                    if 'width' in current_frame and 'height' in current_frame:
                        st.write(f"**Dimensions:** {current_frame['width']} x {current_frame['height']}")
                else:
                    st.warning("No filepath found for current frame")
        else:
            st.info("No video frames found in artifact data")
            st.write(f"Artifact bucket: {current_item.get('artifact_bucket', 'N/A')}")
            st.write(f"Artifact filepath: {current_item.get('artifact_filepath', 'N/A')}")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with main_col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Content")
        
        # Display available content from the new schema and artifact data
        st.markdown("**Classification Reasoning:**")
        reasoning = current_item.get('reasoning', 'No reasoning provided')
        st.write(reasoning)
        
        st.markdown("**Categories:**")
        categories = current_item.get('categories', '{}')
        if categories and categories != '{}':
            try:
                if isinstance(categories, str):
                    categories_data = json.loads(categories)
                else:
                    categories_data = categories
                if categories_data:
                    for key, value in categories_data.items():
                        st.write(f"- {key}: {value}")
                else:
                    st.write("No categories specified")
            except:
                st.write(categories)
        else:
            st.write("No categories specified")
        
        # Display content from artifact data
        if current_artifact_data:
            st.markdown("**Video Content:**")
            
            # Extract text content from artifact
            text_content = current_artifact_data.get('text_content', [])
            
            if isinstance(text_content, list):
                for item in text_content:
                    if isinstance(item, dict):
                        item_type = item.get('type', '')
                        
                        # Handle paragraph content (captions/text)
                        if item_type == 'paragraph':
                            paragraphs = item.get('paragraphs', [])
                            if isinstance(paragraphs, list) and len(paragraphs) > 0:
                                st.markdown("**Video Text/Captions:**")
                                for paragraph in paragraphs:
                                    if isinstance(paragraph, str) and paragraph.strip():
                                        st.markdown("**Original:**")
                                        st.write(paragraph)
                                        
                                        # Add translation for non-English content
                                        try:
                                            detected_lang = detect(paragraph)
                                            if detected_lang != 'en' and detected_lang != 'unknown':
                                                translator = init_translator()
                                                translation = translator.translate(paragraph, dest='en')
                                                st.markdown("**🌐 Translation:**")
                                                st.info(f"({detected_lang} → en): {translation.text}")
                                        except:
                                            pass
                                        st.markdown("---")
                            elif isinstance(paragraphs, str) and paragraphs.strip():
                                st.markdown("**Video Text/Captions:**")
                                st.markdown("**Original:**")
                                st.write(paragraphs)
                                try:
                                    detected_lang = detect(paragraphs)
                                    if detected_lang != 'en' and detected_lang != 'unknown':
                                        translator = init_translator()
                                        translation = translator.translate(paragraphs, dest='en')
                                        st.markdown("**🌐 Translation:**")
                                        st.info(f"({detected_lang} → en): {translation.text}")
                                except:
                                    pass
                                st.markdown("---")
                        
                        # Handle video content
                        elif item_type == 'video':
                            # Check if video data is nested under 'video' key
                            if 'video' in item:
                                video_data = item['video']
                            else:
                                video_data = item
                            
                            if isinstance(video_data, dict):
                                st.markdown("**Video Information:**")
                                st.write(f"🎬 Duration: {video_data.get('duration', 'N/A')} seconds")
                                st.write(f"📐 Dimensions: {video_data.get('width', 'N/A')} x {video_data.get('height', 'N/A')}")
                                
                                # Show video caption if available
                                if 'caption' in video_data and video_data['caption']:
                                    st.markdown("**Video Caption:**")
                                    st.write(video_data['caption'])
                                
                                # Display current frame caption if available
                                if current_frames and current_frame_index < len(current_frames):
                                    current_frame = current_frames[current_frame_index]
                                    frame_caption = current_frame.get('caption', '')
                                    if frame_caption:
                                        st.markdown("**Current Frame Caption:**")
                                        st.info(frame_caption)
        else:
            st.info("No artifact data loaded")
            
        # Display other metadata
        st.markdown("**Additional Information:**")
        st.write(f"Model ID: {current_item.get('model_id', 'N/A')}")
        st.write(f"Classification Mode: {current_item.get('classification_mode', 'N/A')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Add a horizontal line for separation
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    # Content Audit Section
    st.markdown("### 📝 Content Audit")
    
    # Get current frame or content for audit
    current_content_for_audit = "No content available"
    if current_frames and current_frame_index < len(current_frames):
        current_frame = current_frames[current_frame_index]
        current_content_for_audit = current_frame.get('caption', reasoning)
    else:
        current_content_for_audit = reasoning
    
    # Display current content being audited
    content_col1, content_col2 = st.columns([2, 1])
    
    with content_col1:
        st.markdown("**Current Content for Audit:**")
        st.text_area(
            "Content to audit:", 
            value=current_content_for_audit, 
            height=100, 
            disabled=True,
            key=f"{page_key}content_display"
        )
    
    with content_col2:
        st.markdown("**Content Audit Questions:**")
        
        # Content accuracy
        content_accuracy = st.radio(
            "How accurate is this content?",
            ["Very Accurate", "Mostly Accurate", "Partially Accurate", "Inaccurate"],
            index=1,
            key=f"{page_key}content_accuracy"
        )
        
        # Content completeness
        content_completeness = st.radio(
            "How complete is this content?",
            ["Very Complete", "Mostly Complete", "Partially Complete", "Incomplete"],
            index=1,
            key=f"{page_key}content_completeness"
        )
        
        # Content relevance
        content_relevance = st.radio(
            "How relevant is this content?",
            ["Very Relevant", "Mostly Relevant", "Partially Relevant", "Not Relevant"],
            index=1,
            key=f"{page_key}content_relevance"
        )
    
    # Additional content feedback
    st.markdown("**Additional Content Feedback:**")
    content_feedback_col1, content_feedback_col2 = st.columns(2)
    
    with content_feedback_col1:
        content_issues = st.multiselect(
            "Select any issues with this content:",
            [
                "Missing details",
                "Wrong description",
                "Missing context", 
                "Inappropriate content",
                "Too generic/vague",
                "Grammar/spelling errors",
                "Factual errors",
                "Bias detected",
                "Other"
            ],
            key=f"{page_key}content_issues"
        )
    
    with content_feedback_col2:
        improved_content = st.text_area(
            "Suggest improved content (optional):",
            placeholder="Write better content if needed...",
            height=80,
            key=f"{page_key}improved_content"
        )
    
    # Overall content quality score
    st.markdown("**Overall Content Quality Score:**")
    content_quality_score = st.slider(
        "Rate overall content quality (1-10):",
        min_value=1,
        max_value=10,
        value=5,
        key=f"{page_key}content_quality_score"
    )
    
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    # Create three columns for correction sections
    corr_col1, corr_col2, corr_col3 = st.columns(3)

    with corr_col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Data Collection")
        st.write("**Video Data Collection Quality**")
        data_collection_quality = st.radio(
            "Evaluate the quality of video data collection:",
            ["Bad (0)", "Good (1)"],
            index=0,
            key=f"{page_key}data_quality"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with corr_col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Classification Correction")
        st.write("**Video Classification Checking**")
        
        # Dynamic options based on the actual content's classification
        current_classification = current_item.get('common_sense_classification_value', 'N/A')
        if current_classification == 0:
            classification_options = ["Correct - Is CSBS", "Incorrect - Should be Safe"]
        elif current_classification == 100:
            classification_options = ["Correct - Is Safe", "Incorrect - Should be CSBS"]
        else:
            classification_options = ["Floor", "Not Floor"]  # Fallback
        
        classification_check = st.radio(
            "Select the correct classification:",
            classification_options,
            index=0,
            key=f"{page_key}classification"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with corr_col3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Reasoning Quality")
        st.write("**Classification Reasoning Quality**")
        reasoning_quality = st.radio(
            "Evaluate the quality of classification reasoning:",
            ["Bad (0)", "Good (1)"],
            index=0,
            key=f"{page_key}reasoning_quality"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Add buttons directly below the classification section
    st.markdown("<br>", unsafe_allow_html=True)
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        if st.button("Add Correction", key=f"{page_key}add_correction"):
            try:
                # Determine corrected classification based on user selection
                original_classification = current_item.get('common_sense_classification_value')
                if "Should be Safe" in classification_check:
                    corrected_classification = 100
                elif "Should be CSBS" in classification_check:
                    corrected_classification = 0
                else:
                    corrected_classification = original_classification  # No change
                
                correction = {
                    'artifact_id': current_item.get('artifact_id'),
                    'original_classification': original_classification,
                    'page_type': f"Meta Video {page_type}",
                    'corrected_classification': corrected_classification,
                    'correction_value': classification_check,
                    'data_collection_quality': data_collection_quality,
                    'classification_check': classification_check,
                    'reasoning_quality': reasoning_quality,
                    'content_type': 'meta_video',
                    'timestamp': datetime.now().isoformat(),
                    # Content audit data
                    'current_content': current_content_for_audit,
                    'content_accuracy': content_accuracy,
                    'content_completeness': content_completeness,
                    'content_relevance': content_relevance,
                    'content_issues': ', '.join(content_issues) if content_issues else 'None',
                    'improved_content': improved_content.strip() if improved_content.strip() else 'None',
                    'content_quality_score': content_quality_score,
                    'frame_number': current_frame_index + 1 if current_frames else 0,
                    'total_frames': len(current_frames) if current_frames else 0,
                    'artifact_url': current_item.get('artifact_url', ''),
                    'artifact_title': current_item.get('artifact_title', ''),
                    'ai_model_name': current_item.get('ai_model_name', ''),
                    'source': current_item.get('source', ''),
                    'artifact_bucket': current_item.get('artifact_bucket', ''),
                    'artifact_filepath': current_item.get('artifact_filepath', '')
                }
                st.session_state.corrections.append(correction)
                st.success("Meta video correction added!")
                
                # Auto-advance to next record
                if st.session_state[f'{page_key}current_index'] < len(data) - 1:
                    st.session_state[f'{page_key}current_index'] += 1
                    # Reset frame index when advancing
                    st.session_state[f'{page_key}current_frame_index'] = 0
                    st.rerun()
                else:
                    st.info("✅ Reached end of dataset!")
            except Exception as e:
                st.error(f"Error adding correction: {str(e)}")

    with button_col2:
        if st.button("Save Corrections", key=f"{page_key}save_corrections"):
            if st.session_state.corrections:
                try:
                    corrections_df = pd.DataFrame(st.session_state.corrections)
                    st.download_button(
                        label="Download Video Corrections CSV",
                        data=corrections_df.to_csv(index=False),
                        file_name=f"meta_video_corrections_{page_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"{page_key}download"
                    )
                    st.success(f"Saved {len(st.session_state.corrections)} Meta video corrections!")
                except Exception as e:
                    st.error(f"Error saving corrections: {str(e)}")
            else:
                st.warning("No corrections to save")

# Initialize session state
if 'corrections' not in st.session_state:
    st.session_state.corrections = []

# Set page configuration
st.set_page_config(
    page_title="Meta Video Classification Interface",
    page_icon="📱",
    layout="wide"
)

# Sidebar for page navigation
st.sidebar.title("📱 Meta Video Review")
page_selection = st.sidebar.radio(
    "Select Video Content Type:",
    ["🚨 Video CSBS", "✅ Video Safe"],
    index=0
)

# Display corrections count in sidebar
if st.session_state.corrections:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Stats")
    st.sidebar.info(f"Total video corrections: {len(st.session_state.corrections)}")
    
    # Show breakdown by page type
    csbs_corrections = len([c for c in st.session_state.corrections if 'CSBS' in c.get('page_type', '')])
    safe_corrections = len([c for c in st.session_state.corrections if 'Safe' in c.get('page_type', '')])
    
    st.sidebar.write(f"🚨 Video CSBS: {csbs_corrections}")
    st.sidebar.write(f"✅ Video Safe: {safe_corrections}")

# Main content based on page selection
if page_selection == "🚨 Video CSBS":
    display_main_interface("CSBS")
elif page_selection == "✅ Video Safe":
    display_main_interface("Safe")