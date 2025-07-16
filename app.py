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

from logger import logger

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

# OPTIMIZED Function to query BigQuery - Instagram/Facebook videos only with UNIQUE selection
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_cached(date, limit, classification_values):
    """
    OPTIMIZED Query for Instagram and Facebook video content only
    Much faster by pre-filtering for Instagram/Facebook URLs and selecting UNIQUE records
    """
    try:
        client = init_bigquery_client()
        
        if isinstance(classification_values, list):
            classification_conditions = " OR ".join([f"cl.classification_value = {val}" for val in classification_values])
            where_clause = f"({classification_conditions})"
        else:
            where_clause = f"cl.classification_value = {classification_values}"
        
        query = f"""
        WITH unique_content AS (
          SELECT
            cl.artifact_id,
            cl.classification_value,
            cl.classified_at,
            cl.asset_classifications,
            c.artifact,
            meta.text_content,
            ROW_NUMBER() OVER (
              PARTITION BY cl.artifact_id
              ORDER BY cl.classified_at DESC
            ) AS artifact_rn,
            ROW_NUMBER() OVER (
              PARTITION BY c.artifact
              ORDER BY cl.classified_at DESC
            ) AS url_rn
          FROM swift-catfish-337215.meta_brand_safety.content_output_external c
          INNER JOIN swift-catfish-337215.postgres_datastream_aee.public_artifact_common_sense_classification cl
            ON SPLIT(cl.artifact_id,':')[SAFE_OFFSET(1)] = c.content_id
          INNER JOIN swift-catfish-337215.postgres_datastream_aee.public_artifact_metadata meta
            ON cl.artifact_id = meta.artifact_id
          WHERE c.date = "{date}"
          AND {where_clause}
          AND (
            c.artifact LIKE '%instagram.com%' 
            OR c.artifact LIKE '%facebook.com%'
            OR c.artifact LIKE '%fb.com%'
          )
        )
        
        SELECT
          artifact_id,
          classification_value,
          classified_at,
          asset_classifications,
          artifact,
          text_content
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
                df['artifact'].str.contains('/reel/|/video|/videos/', case=False, na=False, regex=True) |
                df['text_content'].str.contains('"type":"video"', case=False, na=False)
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

# OPTIMIZED Function to query BigQuery by Artifact ID - Instagram/Facebook only
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_by_artifact_id(artifact_id, classification_value=None):
    """
    Query BigQuery for a specific Artifact ID - Instagram/Facebook only
    """
    try:
        client = init_bigquery_client()
        
        where_conditions = [f'cl.artifact_id = "{artifact_id}"']
        if classification_value is not None:
            where_conditions.append(f'cl.classification_value = {classification_value}')
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT
        cl.*, c.artifact, meta.text_content
        FROM
         `swift-catfish-337215.postgres_datastream_aee.public_artifact_common_sense_classification` cl
        INNER JOIN
         `swift-catfish-337215.meta_brand_safety.content_output_external` c
        ON
         SPLIT(cl.artifact_id,':')[SAFE_OFFSET(1)] = c.content_id
         inner join `swift-catfish-337215.postgres_datastream_aee.public_artifact_metadata` meta on cl.artifact_id = meta.artifact_id
        WHERE
         {where_clause}
         AND (
           c.artifact LIKE '%instagram.com%'
           OR c.artifact LIKE '%facebook.com%'
           OR c.artifact LIKE '%fb.com%'
         )
        order by classified_at desc
        limit 1
        """
        
        df = client.query(query).to_dataframe()
        logger.debug({"rows": len(df), "artifact_id": artifact_id, "classification_value": classification_value}, 'loaded Instagram/Facebook data by Artifact ID')
        return df
    except Exception as e:
        logger.error({"error": str(e)}, 'failed to load Instagram/Facebook data by Artifact ID')
        raise e

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
    if f'{page_key}current_image_index' not in st.session_state:
        st.session_state[f'{page_key}current_image_index'] = 0

    # Use page-specific state and update local variables
    current_index = st.session_state[f'{page_key}current_index']
    data = st.session_state[f'{page_key}data']
    current_image_index = st.session_state[f'{page_key}current_image_index']

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
        # Cache management
        if st.button("Clear Cache", key=f"{page_key}clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

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
                            data = loaded_data
                            current_index = 0
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
                        data = loaded_data
                        current_index = 0
                        st.success(f"✅ Loaded {len(loaded_data)} unique Meta video records")
                except Exception as e:
                    st.error(f"❌ Error loading video data: {str(e)}")

    # Check if data is loaded
    if data is None or len(data) == 0:
        st.info("Please click 'Load Data' to start reviewing Meta video content.")
        return
    
    # Get current item based on current index
    current_item = data.iloc[current_index].to_dict()

    # Navigation controls
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Previous", key=f"{page_key}prev"):
            if st.session_state[f'{page_key}current_index'] > 0:
                st.session_state[f'{page_key}current_index'] -= 1
                # Reset frame index when changing records
                st.session_state[f'{page_key}current_image_index'] = 0
            st.rerun()

    with col2:
        if st.button("Next", key=f"{page_key}next"):
            if st.session_state[f'{page_key}current_index'] < len(data) - 1:
                st.session_state[f'{page_key}current_index'] += 1
                # Reset frame index when changing records
                st.session_state[f'{page_key}current_image_index'] = 0
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
                st.session_state[f'{page_key}current_image_index'] = 0
                st.rerun()
        else:
            st.write("No records to navigate")

    # Display the current item - navigation and frame reset logic
    # Reset image index when moving to a new record
    if f'{page_key}last_record_index' not in st.session_state or st.session_state[f'{page_key}last_record_index'] != current_index:
        st.session_state[f'{page_key}current_image_index'] = 0
        st.session_state[f'{page_key}last_record_index'] = current_index
        current_image_index = 0
    else:
        current_image_index = st.session_state[f'{page_key}current_image_index']

    # Display metadata with comprehensive debug info
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Metadata")
    
    # Add comprehensive debug info
    full_artifact_id = current_item.get('artifact_id', 'N/A')
    url = current_item.get('artifact', 'N/A')
    classification = current_item.get('classification_value', 'N/A')
    
    st.write(f"🔍 DEBUG: Record Index: {current_index} of {len(data)}")
    st.write(f"🔍 DEBUG: Full Artifact ID: {full_artifact_id}")
    st.write(f"🔍 DEBUG: URL: {url}")
    st.write(f"🔍 DEBUG: Classification: {classification}")
    
    # Show if this is a duplicate by checking previous records
    if current_index > 0:
        prev_urls = []
        for i in range(min(current_index, 5)):  # Check last 5 records
            prev_item = data.iloc[i].to_dict()
            prev_urls.append(prev_item.get('artifact', ''))
        
        if url in prev_urls:
            st.error(f"⚠️ DUPLICATE DETECTED: This URL appeared in previous records!")
        else:
            st.success(f"✅ UNIQUE: This URL is unique in the current dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("ID:", current_item.get('id', 'N/A'))
        st.write("Artifact ID:", full_artifact_id)
        st.write("URL:", url)
    with col2:
        classification_value_display = current_item.get('classification_value', 'N/A')
        if classification_value_display == 0:
            st.markdown('Classification Value: <span style="color: red; font-weight: bold; font-size: 18px;">🚨 Below Common Sense Floor</span>', unsafe_allow_html=True)
        elif classification_value_display == 100:
            st.markdown('Classification Value: <span style="color: green; font-weight: bold; font-size: 18px;">✅ Safe</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'Classification Value: <span style="font-weight: bold; font-size: 18px;">{classification_value_display}</span>', unsafe_allow_html=True)
        st.write("Classified At:", current_item.get('classified_at', 'N/A'))
    with col3:
        # Display content type - emphasize it's Meta video
        artifact_url = current_item.get('artifact', '')
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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)

    # Main content area with two columns
    main_col1, main_col2 = st.columns([1, 1])

    with main_col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Frames")
        try:
            # Get image filepath from asset_classifications
            asset_classifications_raw = current_item.get('asset_classifications', '[]')
            
            # Handle different data types properly
            if isinstance(asset_classifications_raw, str):
                try:
                    asset_classifications = json.loads(asset_classifications_raw)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in asset_classifications")
                    asset_classifications = []
            elif isinstance(asset_classifications_raw, list):
                asset_classifications = asset_classifications_raw
            elif hasattr(asset_classifications_raw, '__iter__') and not isinstance(asset_classifications_raw, str):
                # Handle pandas Series or numpy arrays
                try:
                    asset_classifications = asset_classifications_raw.tolist() if hasattr(asset_classifications_raw, 'tolist') else list(asset_classifications_raw)
                except:
                    asset_classifications = []
            else:
                asset_classifications = []
                
            if asset_classifications and len(asset_classifications) > 0:
                # Filter out classifications that have filepath
                image_classifications = [c for c in asset_classifications if isinstance(c, dict) and c.get('filepath')]
                
                if image_classifications:
                    # Image navigation controls
                    st.markdown("#### Frame Navigation")
                    img_nav_col1, img_nav_col2, img_nav_col3 = st.columns([1, 2, 1])
                    
                    with img_nav_col1:
                        if st.button("⬅️ Previous Frame", key=f"{page_key}prev_frame"):
                            if current_image_index > 0:
                                st.session_state[f'{page_key}current_image_index'] -= 1
                            st.rerun()
                    
                    with img_nav_col2:
                        st.write(f"Frame {current_image_index + 1} of {len(image_classifications)}")
                    
                    with img_nav_col3:
                        if st.button("Next Frame ➡️", key=f"{page_key}next_frame"):
                            if current_image_index < len(image_classifications) - 1:
                                st.session_state[f'{page_key}current_image_index'] += 1
                            st.rerun()
                    
                    # Display current image
                    current_image_classification = image_classifications[current_image_index]
                    image_path = current_image_classification.get('filepath')
                    
                    if image_path and isinstance(image_path, str):
                        # Try different buckets for the image
                        buckets_to_try = [
                            '72025182-c572-6ca6-770e-205857c78546',
                        ]
                        
                        image_loaded = False
                        for bucket_name in buckets_to_try:
                            try:
                                image = load_image_from_gcs(bucket_name, image_path)
                                if image is not None:
                                    st.image(image)
                                    st.info(f"Frame loaded from: {bucket_name}")
                                    image_loaded = True
                                    break
                            except Exception as e:
                                continue
                        
                        if not image_loaded:
                            st.error(f"Could not load frame from any bucket. Path: {image_path}")
                            st.write("Tried buckets:", buckets_to_try)
                            st.write("Frame path:", image_path)
                            
                            # Try alternative path construction
                            try:
                                if 'date=' in image_path and 'hour=' in image_path:
                                    json_bucket = '72025182-c572-6ca6-770e-205857c78546'
                                    if 'artifacts/assets/' in image_path:
                                        json_image_path = f"artifacts/assets/{image_path.split('artifacts/assets/')[-1]}"
                                    else:
                                        json_image_path = f"artifacts/assets/{image_path}"
                                    
                                    st.write(f"Trying alternative path: {json_image_path}")
                                    image = load_image_from_gcs(json_bucket, json_image_path)
                                    if image is not None:
                                        st.image(image)
                                        st.info(f"Frame loaded from alternative path: {json_bucket}")
                                        image_loaded = True
                            except Exception as e:
                                st.write(f"Alternative path attempt failed: {str(e)}")
                else:
                    st.info("No video frames with filepath found in asset_classifications")
            else:
                st.info("No video frame data available")
        except Exception as e:
            st.error(f"Error loading video frames: {str(e)}")
            st.write("Raw asset_classifications:", current_item.get('asset_classifications', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)

    with main_col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Content")
        try:
            # Display text content
            text_content_raw = current_item.get('text_content', '[]')
            
            # Handle different data types properly
            if isinstance(text_content_raw, str):
                try:
                    text_content = json.loads(text_content_raw)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in text_content")
                    text_content = []
            elif isinstance(text_content_raw, list):
                text_content = text_content_raw
            elif hasattr(text_content_raw, '__iter__') and not isinstance(text_content_raw, str):
                # Handle pandas Series or numpy arrays
                try:
                    text_content = text_content_raw.tolist() if hasattr(text_content_raw, 'tolist') else list(text_content_raw)
                except:
                    text_content = []
            else:
                text_content = []
                
            if text_content and len(text_content) > 0:
                for item in text_content:
                    if isinstance(item, dict):
                        item_type = item.get('type', '')
                        
                        # Handle paragraph content (video captions/text)
                        if item_type == 'paragraph':
                            paragraphs = item.get('paragraphs', [])
                            
                            # Handle numpy arrays and other iterable types
                            if hasattr(paragraphs, '__iter__') and not isinstance(paragraphs, str):
                                try:
                                    if hasattr(paragraphs, 'tolist'):
                                        paragraphs = paragraphs.tolist()
                                    else:
                                        paragraphs = list(paragraphs)
                                except:
                                    paragraphs = []
                            
                            if isinstance(paragraphs, list) and len(paragraphs) > 0:
                                st.markdown("**Video Text/Captions:**")
                                for paragraph in paragraphs:
                                    if isinstance(paragraph, str):
                                        # Display original text
                                        st.markdown("**Original:**")
                                        st.write(paragraph)
                                        
                                        # Add translation for non-English content
                                        try:
                                            detected_lang = detect(paragraph)
                                            # Handle potential array return from detect function
                                            if hasattr(detected_lang, '__iter__') and not isinstance(detected_lang, str):
                                                # If it's an array/Series, take the first element
                                                detected_lang = detected_lang[0] if len(detected_lang) > 0 else 'unknown'
                                            
                                            if isinstance(detected_lang, str) and detected_lang != 'en' and detected_lang != 'unknown':
                                                translator = init_translator()
                                                translation = translator.translate(paragraph, dest='en')
                                                st.markdown("**🌐 Translation:**")
                                                st.info(f"({detected_lang} → en): {translation.text}")
                                        except Exception as e:
                                            # Translation failed, continue without it
                                            pass
                                        st.markdown("---")
                            else:
                                st.write("No video text/captions available")
                        
                        # Handle video content
                        elif item_type == 'video':
                            video_data = item.get('video', {})
                            if isinstance(video_data, dict):
                                st.markdown("**Video Information:**")
                                st.write(f"🎬 Duration: {video_data.get('duration', 'N/A')} seconds")
                                st.write(f"📐 Dimensions: {video_data.get('width', 'N/A')} x {video_data.get('height', 'N/A')}")
                                
                                # Display video frames if available
                                frames = video_data.get('frames', [])
                                if isinstance(frames, list) and len(frames) > 0:
                                    st.markdown("**Video Frame Details:**")
                                    for i, frame in enumerate(frames):
                                        if isinstance(frame, dict):
                                            frame_path = frame.get('filepath', '')
                                            if frame_path:
                                                st.write(f"🎞️ Frame {i+1}: {frame_path}")
                                                st.write(f"⏱️ Timestamp: {frame.get('timestamp', 'N/A')}ms")
                                                st.write(f"📏 Size: {frame.get('width', 'N/A')} x {frame.get('height', 'N/A')}")
                                                st.markdown("---")
            
            # Display image caption from asset_classifications
            asset_classifications_raw = current_item.get('asset_classifications', '[]')
            
            # Handle different data types properly
            if isinstance(asset_classifications_raw, str):
                try:
                    asset_classifications = json.loads(asset_classifications_raw)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in asset_classifications")
                    asset_classifications = []
            elif isinstance(asset_classifications_raw, list):
                asset_classifications = asset_classifications_raw
            elif hasattr(asset_classifications_raw, '__iter__') and not isinstance(asset_classifications_raw, str):
                # Handle pandas Series or numpy arrays
                try:
                    asset_classifications = asset_classifications_raw.tolist() if hasattr(asset_classifications_raw, 'tolist') else list(asset_classifications_raw)
                except:
                    asset_classifications = []
            else:
                asset_classifications = []
                
            if asset_classifications and len(asset_classifications) > 0:
                # Filter out classifications that have filepath (same as image section)
                image_classifications = [c for c in asset_classifications if isinstance(c, dict) and c.get('filepath')]
                
                if image_classifications and current_image_index < len(image_classifications):
                    # Display caption for current frame only
                    current_image_classification = image_classifications[current_image_index]
                    caption = current_image_classification.get('caption', 'No caption available')
                    if isinstance(caption, str):
                        st.markdown("**Frame Caption:**")
                        st.write(caption)
                        st.markdown("---")
            
            if not text_content or len(text_content) == 0:
                st.info("No video content available")
        except Exception as e:
            st.error(f"Error parsing video content: {str(e)}")
            st.write("Raw text_content:", current_item.get('text_content', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)

    # Add a horizontal line for separation
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    # Caption Audit Section
    st.markdown("### 📝 Caption Audit")
    
    # Get current frame caption for audit
    current_frame_caption = "No caption available"
    if asset_classifications and len(asset_classifications) > 0:
        image_classifications = [c for c in asset_classifications if isinstance(c, dict) and c.get('filepath')]
        if image_classifications and current_image_index < len(image_classifications):
            current_image_classification = image_classifications[current_image_index]
            current_frame_caption = current_image_classification.get('caption', 'No caption available')
    
    # Display current caption being audited
    caption_col1, caption_col2 = st.columns([2, 1])
    
    with caption_col1:
        st.markdown("**Current Frame Caption:**")
        st.text_area(
            "Caption to audit:", 
            value=current_frame_caption, 
            height=100, 
            disabled=True,
            key=f"{page_key}caption_display"
        )
    
    with caption_col2:
        st.markdown("**Caption Audit Questions:**")
        
        # Caption accuracy
        caption_accuracy = st.radio(
            "How accurate is this caption?",
            ["Very Accurate", "Mostly Accurate", "Partially Accurate", "Inaccurate"],
            index=1,
            key=f"{page_key}caption_accuracy"
        )
        
        # Caption completeness
        caption_completeness = st.radio(
            "How complete is this caption?",
            ["Very Complete", "Mostly Complete", "Partially Complete", "Incomplete"],
            index=1,
            key=f"{page_key}caption_completeness"
        )
        
        # Caption relevance
        caption_relevance = st.radio(
            "How relevant is this caption?",
            ["Very Relevant", "Mostly Relevant", "Partially Relevant", "Not Relevant"],
            index=1,
            key=f"{page_key}caption_relevance"
        )
    
    # Additional caption feedback
    st.markdown("**Additional Caption Feedback:**")
    caption_feedback_col1, caption_feedback_col2 = st.columns(2)
    
    with caption_feedback_col1:
        caption_issues = st.multiselect(
            "Select any issues with this caption:",
            [
                "Missing objects/people",
                "Wrong object description",
                "Missing actions/activities", 
                "Wrong scene description",
                "Missing text/overlays",
                "Inappropriate content described",
                "Too generic/vague",
                "Grammar/spelling errors",
                "Other"
            ],
            key=f"{page_key}caption_issues"
        )
    
    with caption_feedback_col2:
        improved_caption = st.text_area(
            "Suggest improved caption (optional):",
            placeholder="Write a better caption if needed...",
            height=80,
            key=f"{page_key}improved_caption"
        )
    
    # Overall caption quality score
    st.markdown("**Overall Caption Quality Score:**")
    caption_quality_score = st.slider(
        "Rate overall caption quality (1-10):",
        min_value=1,
        max_value=10,
        value=5,
        key=f"{page_key}caption_quality_score"
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
        current_classification = current_item.get('classification_value', 'N/A')
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
        st.markdown("### Scene Detection Quality")
        st.write("**Video Scene Detection Quality**")
        image_caption_quality = st.radio(
            "Evaluate the quality of video scene detection:",
            ["Bad (0)", "Good (1)"],
            index=0,
            key=f"{page_key}caption_quality"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Add buttons directly below the classification section
    st.markdown("<br>", unsafe_allow_html=True)
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        if st.button("Add Correction", key=f"{page_key}add_correction"):
            try:
                # Determine corrected classification based on user selection
                original_classification = current_item.get('classification_value')
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
                    'image_caption_quality': image_caption_quality,
                    'content_type': 'meta_video',
                    'timestamp': datetime.now().isoformat(),
                    # Caption audit data
                    'current_frame_caption': current_frame_caption,
                    'caption_accuracy': caption_accuracy,
                    'caption_completeness': caption_completeness,
                    'caption_relevance': caption_relevance,
                    'caption_issues': ', '.join(caption_issues) if caption_issues else 'None',
                    'improved_caption': improved_caption.strip() if improved_caption.strip() else 'None',
                    'caption_quality_score': caption_quality_score,
                    'frame_number': current_image_index + 1
                }
                st.session_state.corrections.append(correction)
                st.success("Meta video correction added!")
                
                # Auto-advance to next record
                if st.session_state[f'{page_key}current_index'] < len(data) - 1:
                    st.session_state[f'{page_key}current_index'] += 1
                    # Reset frame index when advancing
                    st.session_state[f'{page_key}current_image_index'] = 0
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