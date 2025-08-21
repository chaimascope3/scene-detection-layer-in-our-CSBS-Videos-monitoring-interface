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
import requests

from logger import logger

# Initialize BLIP model for image captioning (Large version)
@st.cache_resource
def init_blip_model():
    """Initialize BLIP large model for image captioning"""
    try:
        # Use BLIP large model (original BLIP, not BLIP-2)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        return processor, model
    except Exception as e:
        st.error(f"Error loading BLIP model: {str(e)}")
        return None, None

# Function to generate image caption using BLIP large
def generate_blip_caption(image, processor=None, model=None):
    """Generate image caption using BLIP large model"""
    try:
        if processor is None or model is None:
            processor, model = init_blip_model()
            if processor is None or model is None:
                return "BLIP model not available"
        
        # Prepare image
        if isinstance(image, str):
            # If image is a URL or path, try to load it
            if image.startswith('http'):
                response = requests.get(image)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image)
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image and generate caption
        inputs = processor(image, return_tensors="pt")
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate caption with BLIP
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption.strip()
        
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# Enhanced function to try loading image and generate caption if missing
def try_load_image_with_blip_caption(image_path, crawler_bucket, crawler_asset_filepaths, existing_caption=None):
    """Try to load an image and generate BLIP caption if none exists"""
    try:
        # First try to load the image
        image = None
        bucket_used = None
        path_used = None
        
        # Build list of buckets to try
        buckets_to_try = []
        if crawler_bucket and isinstance(crawler_bucket, str) and crawler_bucket.strip():
            buckets_to_try.append(crawler_bucket.strip())
        
        # Add default buckets
        buckets_to_try.extend([
            '72025182-c572-6ca6-770e-205857c78546',
            'scope3-prod'
        ])
        
        # Remove duplicates while preserving order
        buckets_to_try = list(dict.fromkeys(buckets_to_try))
        
        # Try different path variations
        path_variations = [image_path]
        if not image_path.startswith('artifacts/assets/'):
            path_variations.append(f"artifacts/assets/{image_path}")
        
        # Try enhanced path from crawler filepaths
        enhanced_path = get_image_path(None, None, image_path, crawler_asset_filepaths)
        if enhanced_path and enhanced_path not in path_variations:
            path_variations.append(enhanced_path)
        
        # Try to load image from different sources
        for bucket_name in buckets_to_try:
            for path_var in path_variations:
                try:
                    image = load_image_from_gcs(bucket_name, path_var, crawler_bucket)
                    if image is not None:
                        bucket_used = bucket_name
                        path_used = path_var
                        break
                except Exception:
                    continue
            if image is not None:
                break
        
        if image is not None:
            # Display the image
            st.image(image)
            st.info(f"✅ Image loaded from: **{bucket_used}** | Path: **{path_used}**")
            
            # Check if we need to generate a caption
            caption_info = []
            
            if existing_caption and existing_caption != "No caption available":
                caption_info.append(f"**Existing Caption:** {existing_caption}")
            else:
                caption_info.append("**Existing Caption:** None")
            
            # Always try to generate BLIP caption for comparison
            with st.spinner("🤖 Generating BLIP Large caption..."):
                try:
                    processor, model = init_blip_model()
                    if processor is not None and model is not None:
                        blip_caption = generate_blip_caption(image, processor, model)
                        caption_info.append(f"**BLIP Large Generated:** {blip_caption}")
                        
                        # Store BLIP caption in session state for use in audit section
                        if 'blip_caption_session' not in st.session_state:
                            st.session_state.blip_caption_session = {}
                        
                        # Create a unique key for this image
                        page_key = st.session_state.get('current_page_key', 'default')
                        current_index = st.session_state.get(f'{page_key}current_index', 0)
                        current_image_index = st.session_state.get(f'{page_key}current_image_index', 0)
                        session_key = f"{page_key}_{current_index}_{current_image_index}"
                        st.session_state.blip_caption_session[session_key] = blip_caption
                        
                        # Show caption comparison
                        st.markdown("### 📝 Caption Analysis")
                        for info in caption_info:
                            st.write(info)
                        
                        # If no existing caption, highlight the BLIP caption
                        if not existing_caption or existing_caption == "No caption available":
                            st.success("🆕 BLIP Large caption generated for frame without description!")
                        else:
                            st.info("ℹ️ BLIP Large caption generated for comparison")
                    else:
                        st.warning("BLIP Large model not available for caption generation")
                except Exception as e:
                    st.error(f"Error generating BLIP caption: {str(e)}")
            
            return True
        else:
            # Log what we tried
            st.error(f"❌ Failed to load image from any source:")
            st.write(f"**Tried buckets**: {buckets_to_try}")
            st.write(f"**Tried paths**: {path_variations}")
            return False
        
    except Exception as e:
        st.error(f"Error trying to load image with BLIP caption: {str(e)}")
        return False

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

# Helper function to check if a path is likely an image file
def is_image_file(filepath):
    """Check if a filepath appears to be an image file"""
    if not isinstance(filepath, str):
        return False
    return any(ext in filepath.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'])

# Helper function to parse filepath data
def parse_filepath_data(filepath_data):
    """Parse various formats of filepath data"""
    try:
        if isinstance(filepath_data, str):
            try:
                # Try JSON parsing first
                return json.loads(filepath_data)
            except:
                # If not JSON, treat as single path
                return [filepath_data]
        elif isinstance(filepath_data, list):
            return filepath_data
        elif hasattr(filepath_data, '__iter__') and not isinstance(filepath_data, str):
            try:
                return filepath_data.tolist() if hasattr(filepath_data, 'tolist') else list(filepath_data)
            except:
                return []
        else:
            return []
    except:
        return []

# Helper function to extract images from content data
def extract_images_from_content(content_data):
    """Extract potential image references from content data"""
    image_refs = []
    
    def search_dict_for_images(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and is_image_file(value):
                    image_refs.append(value)
                elif key.lower() in ['filepath', 'path', 'image_path', 'url', 'image_url'] and isinstance(value, str):
                    image_refs.append(value)
                elif isinstance(value, (dict, list)):
                    search_dict_for_images(value, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                search_dict_for_images(item, f"{path}[{i}]")
    
    try:
        search_dict_for_images(content_data)
        return list(set(image_refs))  # Remove duplicates
    except:
        return []

# Helper function to try extracting images from any data structure
def try_extract_images_from_data(data, crawler_bucket, data_source_name):
    """Try to extract and load images from any data structure"""
    st.write(f"**Searching for images in {data_source_name}:**")
    
    image_refs = extract_images_from_content(data)
    if image_refs:
        st.write(f"Found {len(image_refs)} potential image references:")
        for img_ref in image_refs:
            st.write(f"- {img_ref}")
            load_success = try_load_image_from_multiple_sources(img_ref, crawler_bucket, None)
            if load_success:
                return True
    else:
        st.write(f"No image references found in {data_source_name}")
    return False

# Helper function to try loading image from multiple sources and buckets
def try_load_image_from_multiple_sources(image_path, crawler_bucket, crawler_asset_filepaths):
    """Try to load an image from multiple bucket sources"""
    try:
        # Build list of buckets to try
        buckets_to_try = []
        if crawler_bucket and isinstance(crawler_bucket, str) and crawler_bucket.strip():
            buckets_to_try.append(crawler_bucket.strip())
        
        # Add default buckets
        buckets_to_try.extend([
            '72025182-c572-6ca6-770e-205857c78546',
            'scope3-prod'
        ])
        
        # Remove duplicates while preserving order
        buckets_to_try = list(dict.fromkeys(buckets_to_try))
        
        # Try different path variations
        path_variations = [image_path]
        if not image_path.startswith('artifacts/assets/'):
            path_variations.append(f"artifacts/assets/{image_path}")
        
        # Try enhanced path from crawler filepaths
        enhanced_path = get_image_path(None, None, image_path, crawler_asset_filepaths)
        if enhanced_path and enhanced_path not in path_variations:
            path_variations.append(enhanced_path)
        
        for bucket_name in buckets_to_try:
            for path_var in path_variations:
                try:
                    image = load_image_from_gcs(bucket_name, path_var, crawler_bucket)
                    if image is not None:
                        st.image(image)
                        st.info(f"✅ Image loaded from: **{bucket_name}** | Path: **{path_var}**")
                        return True
                except Exception:
                    continue
        
        # Log what we tried
        st.error(f"❌ Failed to load image from any source:")
        st.write(f"**Tried buckets**: {buckets_to_try}")
        st.write(f"**Tried paths**: {path_variations}")
        return False
        
    except Exception as e:
        st.error(f"Error trying to load image: {str(e)}")
        return False

# Helper function to parse filepath data
def parse_filepath_data(filepath_data):
    """Parse various formats of filepath data"""
    try:
        if isinstance(filepath_data, str):
            try:
                # Try JSON parsing first
                return json.loads(filepath_data)
            except:
                # If not JSON, treat as single path
                return [filepath_data]
        elif isinstance(filepath_data, list):
            return filepath_data
        elif hasattr(filepath_data, '__iter__') and not isinstance(filepath_data, str):
            try:
                return filepath_data.tolist() if hasattr(filepath_data, 'tolist') else list(filepath_data)
            except:
                return []
        else:
            return []
    except:
        return []

# Helper function to extract images from content data
def extract_images_from_content(content_data):
    """Extract potential image references from content data"""
    image_refs = []
    
    def search_dict_for_images(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and is_image_file(value):
                    image_refs.append(value)
                elif key.lower() in ['filepath', 'path', 'image_path', 'url', 'image_url'] and isinstance(value, str):
                    image_refs.append(value)
                elif isinstance(value, (dict, list)):
                    search_dict_for_images(value, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                search_dict_for_images(item, f"{path}[{i}]")
    
    try:
        search_dict_for_images(content_data)
        return list(set(image_refs))  # Remove duplicates
    except:
        return []

# Helper function to try extracting images from any data structure
def try_extract_images_from_data(data, crawler_bucket, data_source_name):
    """Try to extract and load images from any data structure"""
    st.write(f"**Searching for images in {data_source_name}:**")
    
    image_refs = extract_images_from_content(data)
    if image_refs:
        st.write(f"Found {len(image_refs)} potential image references:")
        for img_ref in image_refs:
            st.write(f"- {img_ref}")
            load_success = try_load_image_from_multiple_sources(img_ref, crawler_bucket, None)
            if load_success:
                return True
    else:
        st.write(f"No image references found in {data_source_name}")
    return False

# Helper function to try loading image from multiple sources and buckets
def try_load_image_from_multiple_sources(image_path, crawler_bucket, crawler_asset_filepaths):
    """Try to load an image from multiple bucket sources"""
    try:
        # Build list of buckets to try
        buckets_to_try = []
        if crawler_bucket and isinstance(crawler_bucket, str) and crawler_bucket.strip():
            buckets_to_try.append(crawler_bucket.strip())
        
        # Add default buckets
        buckets_to_try.extend([
            '72025182-c572-6ca6-770e-205857c78546',
            'scope3-prod'
        ])
        
        # Remove duplicates while preserving order
        buckets_to_try = list(dict.fromkeys(buckets_to_try))
        
        # Try different path variations
        path_variations = [image_path]
        if not image_path.startswith('artifacts/assets/'):
            path_variations.append(f"artifacts/assets/{image_path}")
        
        # Try enhanced path from crawler filepaths
        enhanced_path = get_image_path(None, None, image_path, crawler_asset_filepaths)
        if enhanced_path and enhanced_path not in path_variations:
            path_variations.append(enhanced_path)
        
        for bucket_name in buckets_to_try:
            for path_var in path_variations:
                try:
                    image = load_image_from_gcs(bucket_name, path_var, crawler_bucket)
                    if image is not None:
                        st.image(image)
                        st.info(f"✅ Image loaded from: **{bucket_name}** | Path: **{path_var}**")
                        return True
                except Exception:
                    continue
        
        # Log what we tried
        st.error(f"❌ Failed to load image from any source:")
        st.write(f"**Tried buckets**: {buckets_to_try}")
        st.write(f"**Tried paths**: {path_variations}")
        return False
        
    except Exception as e:
        st.error(f"Error trying to load image: {str(e)}")
        return False

# OPTIMIZED Function to query BigQuery - Instagram/Facebook videos only with UNIQUE selection
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_cached(date, limit, classification_values):
    """
    OPTIMIZED Query using both tables:
    - public_artifact_common_sense_classification (for classification data)
    - public_artifact_metadata (for content metadata)
    Much faster by pre-filtering for Instagram/Facebook URLs and selecting UNIQUE records
    """
    try:
        client = init_bigquery_client()
        
        # Handle classification values properly to avoid mixing list and non-list
        if isinstance(classification_values, list):
            if len(classification_values) == 1:
                # Single value case
                where_clause = f"cl.classification_value = {classification_values[0]}"
            else:
                # Multiple values case - use IN operator
                classification_list = ', '.join([str(val) for val in classification_values])
                where_clause = f"cl.classification_value IN ({classification_list})"
        else:
            # Single value case
            where_clause = f"cl.classification_value = {classification_values}"
        
        query = f"""
        WITH unique_content AS (
          SELECT
            cl.artifact_id,
            cl.classification_value,
            cl.classified_at,
            cl.asset_classifications,
            cl.artifact_lang,
            c.artifact,
            meta.text_content,
            meta.crawler_output_bucket,
            meta.crawler_artifact_filepath,
            meta.crawler_asset_filepaths,
            meta.screenshot_filepath,
            meta.metadata as content_metadata,
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
          artifact_lang,
          artifact,
          text_content,
          crawler_output_bucket,
          crawler_artifact_filepath,
          crawler_asset_filepaths,
          screenshot_filepath,
          content_metadata
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
        
        logger.debug({"rows": len(df), "date": date, "classification_values": classification_values}, 'loaded unique Instagram/Facebook video data from both tables')
        return df
    except Exception as e:
        logger.error({"error": str(e)}, 'failed to load unique Instagram/Facebook video data from both tables')
        raise e

# Function to get image path based on bucket structure - ENHANCED with metadata table info
def get_image_path(bucket_name, json_path, image_identifier, crawler_filepaths=None):
    try:
        # Check if we have crawler asset filepaths from metadata table
        if crawler_filepaths is not None:
            # Handle pandas Series or numpy arrays
            if hasattr(crawler_filepaths, '__iter__') and not isinstance(crawler_filepaths, str):
                try:
                    if hasattr(crawler_filepaths, 'tolist'):
                        crawler_filepaths = crawler_filepaths.tolist()
                    else:
                        crawler_filepaths = list(crawler_filepaths)
                except:
                    crawler_filepaths = None
            
            # Handle string (JSON) format
            if isinstance(crawler_filepaths, str):
                try:
                    crawler_filepaths = json.loads(crawler_filepaths)
                except:
                    crawler_filepaths = None
            
            # Use the first filepath if we have a valid list
            if isinstance(crawler_filepaths, list) and len(crawler_filepaths) > 0:
                return crawler_filepaths[0]
        
        # If the image identifier is a full GCS path, extract just the path part
        if isinstance(image_identifier, str) and image_identifier.startswith('gs://'):
            match = re.match(r'gs://[^/]+/(.+)', image_identifier)
            if match:
                return match.group(1)
        
        # If the image identifier already has the full path, use it as is
        if isinstance(image_identifier, str) and image_identifier.startswith('artifacts/assets/'):
            return image_identifier
            
        # Otherwise, construct the path
        if isinstance(image_identifier, str):
            return f"artifacts/assets/{image_identifier}"
        else:
            return None
    except Exception as e:
        st.error(f"Error constructing image path: {str(e)}")
        return None

# Function to read JSON from GCS - ENHANCED with metadata table bucket info
def read_json_from_gcs(bucket_name, json_path, crawler_output_bucket=None):
    try:
        client = init_gcs_client()
        
        # Try using crawler_output_bucket from metadata table first
        if crawler_output_bucket is not None and isinstance(crawler_output_bucket, str) and crawler_output_bucket.strip():
            bucket = client.bucket(crawler_output_bucket)
        else:
            bucket = client.bucket(bucket_name)
            
        blob = bucket.blob(json_path)

        # Check if blob exists
        if not blob.exists():
            st.error(f"JSON file not found: gs://{bucket.name}/{json_path}")
            return None

        content = blob.download_as_string()
        json_data = json.loads(content.decode('utf-8'))
        return json_data
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return None

# Function to load image from GCS - ENHANCED with metadata table bucket info
def load_image_from_gcs(bucket_name, image_path, crawler_output_bucket=None):
    try:
        client = init_gcs_client()
        
        # Try using crawler_output_bucket from metadata table first
        if crawler_output_bucket is not None and isinstance(crawler_output_bucket, str) and crawler_output_bucket.strip():
            bucket = client.bucket(crawler_output_bucket)
        else:
            bucket = client.bucket(bucket_name)

        blob = bucket.blob(image_path)

        # Check if blob exists
        if not blob.exists():
            st.error(f"Image file not found: gs://{bucket.name}/{image_path}")
            return None

        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# OPTIMIZED Function to query BigQuery by Artifact ID - using both tables
@st.cache_data(ttl=7200)  # Cache for 2 hours
def query_bigquery_by_artifact_id(artifact_id, classification_value=None):
    """
    Query BigQuery for a specific Artifact ID using both tables
    """
    try:
        client = init_bigquery_client()
        
        where_conditions = [f'cl.artifact_id = "{artifact_id}"']
        if classification_value is not None:
            where_conditions.append(f'cl.classification_value = {classification_value}')
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT
        cl.artifact_id,
        cl.classification_value,
        cl.classified_at,
        cl.asset_classifications,
        cl.artifact_lang,
        cl.classification_requested_at,
        cl.first_seen_at,
        c.artifact,
        meta.text_content,
        meta.crawler_output_bucket,
        meta.crawler_artifact_filepath,
        meta.crawler_asset_filepaths,
        meta.screenshot_filepath,
        meta.metadata as content_metadata,
        meta.created_at as metadata_created_at,
        meta.updated_at as metadata_updated_at
        FROM
         `swift-catfish-337215.postgres_datastream_aee.public_artifact_common_sense_classification` cl
        INNER JOIN
         `swift-catfish-337215.meta_brand_safety.content_output_external` c
        ON
         SPLIT(cl.artifact_id,':')[SAFE_OFFSET(1)] = c.content_id
        INNER JOIN 
         `swift-catfish-337215.postgres_datastream_aee.public_artifact_metadata` meta 
        ON 
         cl.artifact_id = meta.artifact_id
        WHERE
         {where_clause}
         AND (
           c.artifact LIKE '%instagram.com%'
           OR c.artifact LIKE '%facebook.com%'
           OR c.artifact LIKE '%fb.com%'
         )
        ORDER BY classified_at DESC
        LIMIT 1
        """
        
        df = client.query(query).to_dataframe()
        logger.debug({"rows": len(df), "artifact_id": artifact_id, "classification_value": classification_value}, 'loaded Instagram/Facebook data by Artifact ID from both tables')
        return df
    except Exception as e:
        logger.error({"error": str(e)}, 'failed to load Instagram/Facebook data by Artifact ID from both tables')
        raise e

# Function to display the main interface content - ENHANCED with metadata table fields
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
                            # Query each classification separately to avoid mixing list and non-list values
                            st.info("Loading both CSBS (0) and Safe (100) classifications...")
                            
                            # Load CSBS data (classification_value = 0)
                            csbs_data = query_bigquery_cached(selected_date.strftime('%Y-%m-%d'), effective_limit // 2, 0)
                            
                            # Load Safe data (classification_value = 100)  
                            safe_data = query_bigquery_cached(selected_date.strftime('%Y-%m-%d'), effective_limit // 2, 100)
                            
                            # Combine and sort the results
                            if len(csbs_data) > 0 and len(safe_data) > 0:
                                loaded_data = pd.concat([csbs_data, safe_data], ignore_index=True)
                            elif len(csbs_data) > 0:
                                loaded_data = csbs_data
                            elif len(safe_data) > 0:
                                loaded_data = safe_data
                            else:
                                loaded_data = pd.DataFrame()
                            
                            # Sort by classified_at and limit
                            if len(loaded_data) > 0:
                                loaded_data = loaded_data.sort_values('classified_at', ascending=False).head(effective_limit)
                        else:
                            # Use single classification value
                            loaded_data = query_bigquery_cached(selected_date.strftime('%Y-%m-%d'), effective_limit, selected_classification)
                        
                        st.session_state[f'{page_key}data'] = loaded_data
                        st.session_state[f'{page_key}current_index'] = 0
                        data = loaded_data
                        current_index = 0
                        st.success(f"✅ Loaded {len(loaded_data)} unique Meta video records from both tables")
                except Exception as e:
                    st.error(f"❌ Error loading video data: {str(e)}")
                    # Additional debugging info
                    st.error(f"Classification selected: {selected_classification}")
                    st.error(f"Date: {selected_date}")
                    st.error(f"Limit: {effective_limit}")

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

    # Display metadata with comprehensive debug info - ENHANCED with metadata table fields
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Metadata")
    
    # Add comprehensive debug info
    full_artifact_id = current_item.get('artifact_id', 'N/A')
    url = current_item.get('artifact', 'N/A')
    classification = current_item.get('classification_value', 'N/A')
    crawler_bucket = current_item.get('crawler_output_bucket', 'N/A')
    
    st.write(f"🔍 DEBUG: Record Index: {current_index} of {len(data)}")
    st.write(f"🔍 DEBUG: Full Artifact ID: {full_artifact_id}")
    st.write(f"🔍 DEBUG: URL: {url}")
    st.write(f"🔍 DEBUG: Classification: {classification}")
    st.write(f"🔍 DEBUG: Crawler Bucket: {crawler_bucket}")
    
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
        st.write("Artifact ID:", full_artifact_id)
        st.write("URL:", url)
        st.write("Crawler Bucket:", crawler_bucket)
    with col2:
        classification_value_display = current_item.get('classification_value', 'N/A')
        if classification_value_display == 0:
            st.markdown('Classification Value: <span style="color: red; font-weight: bold; font-size: 18px;">🚨 Below Common Sense Floor</span>', unsafe_allow_html=True)
        elif classification_value_display == 100:
            st.markdown('Classification Value: <span style="color: green; font-weight: bold; font-size: 18px;">✅ Safe</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'Classification Value: <span style="font-weight: bold; font-size: 18px;">{classification_value_display}</span>', unsafe_allow_html=True)
        st.write("Classified At:", current_item.get('classified_at', 'N/A'))
        st.write("Metadata Created:", current_item.get('metadata_created_at', 'N/A'))
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
        st.write("Screenshot Path:", current_item.get('screenshot_filepath', 'N/A'))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)

    # Main content area with two columns
    main_col1, main_col2 = st.columns([1, 1])

    with main_col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Frames")
        
        try:
            # ENHANCED: Comprehensive image search across all available fields
            st.markdown("#### 🔍 Comprehensive Image Search")
            
            # Collect all potential image sources
            image_sources = {}
            
            # Helper function to safely check if a value is non-empty
            def is_non_empty_value(value):
                try:
                    # Handle None
                    if value is None:
                        return False
                    
                    # Handle pandas Series or numpy arrays
                    if hasattr(value, '__len__') and hasattr(value, '__iter__') and not isinstance(value, str):
                        try:
                            # Convert to string first for checking
                            if hasattr(value, 'iloc') and len(value) > 0:
                                # It's a pandas Series, get first value
                                value = str(value.iloc[0]) if len(value) > 0 else ''
                            elif hasattr(value, 'tolist'):
                                # It's a numpy array
                                value_list = value.tolist()
                                value = str(value_list[0]) if len(value_list) > 0 else ''
                            else:
                                # Try to get first element
                                value = str(list(value)[0]) if len(list(value)) > 0 else ''
                        except:
                            return False
                    
                    # Now check the string value
                    value_str = str(value).strip()
                    return value_str not in ['[]', 'null', 'None', '', 'nan', 'NaN']
                except:
                    return False
            
            # 1. asset_classifications
            asset_classifications_raw = current_item.get('asset_classifications', '[]')
            if is_non_empty_value(asset_classifications_raw):
                image_sources['asset_classifications'] = asset_classifications_raw
            
            # 2. screenshot_filepath
            screenshot_path = current_item.get('screenshot_filepath')
            if is_non_empty_value(screenshot_path):
                image_sources['screenshot_filepath'] = screenshot_path
            
            # 3. crawler_asset_filepaths
            crawler_asset_filepaths = current_item.get('crawler_asset_filepaths')
            if is_non_empty_value(crawler_asset_filepaths):
                image_sources['crawler_asset_filepaths'] = crawler_asset_filepaths
            
            # 4. crawler_artifact_filepath
            crawler_artifact_filepath = current_item.get('crawler_artifact_filepath')
            if is_non_empty_value(crawler_artifact_filepath):
                image_sources['crawler_artifact_filepath'] = crawler_artifact_filepath
            
            # 5. Check text_content for embedded image references
            text_content_raw = current_item.get('text_content', '[]')
            if is_non_empty_value(text_content_raw):
                image_sources['text_content'] = text_content_raw
            
            # 6. Check content_metadata for image references
            content_metadata_raw = current_item.get('content_metadata')
            if is_non_empty_value(content_metadata_raw):
                image_sources['content_metadata'] = content_metadata_raw
            
            crawler_bucket = current_item.get('crawler_output_bucket')
            
            # DEBUG: Show what sources we found
            with st.expander("🔍 Available Image Sources", expanded=True):
                st.write(f"**Found {len(image_sources)} potential image sources:**")
                for source_name, source_data in image_sources.items():
                    # Safely display source data
                    try:
                        if hasattr(source_data, '__len__') and hasattr(source_data, '__iter__') and not isinstance(source_data, str):
                            # Handle pandas Series or numpy arrays
                            if hasattr(source_data, 'iloc') and len(source_data) > 0:
                                display_data = str(source_data.iloc[0])[:100]
                            elif hasattr(source_data, 'tolist'):
                                display_data = str(source_data.tolist()[:1])[:100] 
                            else:
                                display_data = str(list(source_data)[:1])[:100]
                        else:
                            display_data = str(source_data)[:100]
                        st.write(f"- **{source_name}**: {type(source_data)} - {display_data}...")
                    except:
                        st.write(f"- **{source_name}**: {type(source_data)} - [Could not display]")
                st.write(f"**Crawler bucket**: {crawler_bucket}")
            
            images_loaded = False
            
            # Try each source in priority order
            
            # Priority 1: asset_classifications (original approach)
            if 'asset_classifications' in image_sources:
                st.markdown("#### 📊 Asset Classifications")
                asset_classifications_raw = image_sources['asset_classifications']
                
                # Handle different data types properly
                asset_classifications = []
                if isinstance(asset_classifications_raw, str):
                    try:
                        asset_classifications = json.loads(asset_classifications_raw)
                    except json.JSONDecodeError as e:
                        st.write(f"JSON parse error: {str(e)}")
                elif isinstance(asset_classifications_raw, list):
                    asset_classifications = asset_classifications_raw
                elif hasattr(asset_classifications_raw, '__iter__') and not isinstance(asset_classifications_raw, str):
                    try:
                        asset_classifications = asset_classifications_raw.tolist() if hasattr(asset_classifications_raw, 'tolist') else list(asset_classifications_raw)
                    except:
                        pass
                
                if asset_classifications and len(asset_classifications) > 0:
                    # Look for items with filepath
                    image_classifications = [c for c in asset_classifications if isinstance(c, dict) and c.get('filepath')]
                    
                    if image_classifications:
                        images_loaded = True
                        st.success(f"Found {len(image_classifications)} images in asset_classifications")
                        
                        # Image navigation controls
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
                        
                        # Display current image with BLIP caption generation
                        current_image_classification = image_classifications[current_image_index]
                        image_path = current_image_classification.get('filepath')
                        existing_caption = current_image_classification.get('caption', 'No caption available')
                        
                        if image_path and isinstance(image_path, str):
                            load_success = try_load_image_with_blip_caption(
                                image_path, 
                                crawler_bucket, 
                                crawler_asset_filepaths,
                                existing_caption
                            )
                            if not load_success:
                                st.error(f"Could not load frame: {image_path}")
                    else:
                        st.write("No items with 'filepath' found in asset_classifications")
                        # Try to extract any image references from asset_classifications
                        try_extract_images_from_data(asset_classifications, crawler_bucket, "asset_classifications items")
                else:
                    st.write("asset_classifications is empty after parsing")
            
            # Priority 2: screenshot_filepath with BLIP caption
            if not images_loaded and 'screenshot_filepath' in image_sources:
                st.markdown("#### 📸 Screenshot")
                screenshot_path = image_sources['screenshot_filepath']
                st.write(f"**Screenshot path**: {screenshot_path}")
                
                load_success = try_load_image_with_blip_caption(screenshot_path, crawler_bucket, None, "No caption available")
                if load_success:
                    images_loaded = True
                else:
                    st.error(f"Could not load screenshot: {screenshot_path}")
            
            # Priority 3: crawler_asset_filepaths
            if not images_loaded and 'crawler_asset_filepaths' in image_sources:
                st.markdown("#### 📁 Crawler Asset Files")
                crawler_filepaths_raw = image_sources['crawler_asset_filepaths']
                st.write(f"**Raw crawler filepaths**: {crawler_filepaths_raw}")
                
                # Parse crawler asset filepaths
                parsed_filepaths = parse_filepath_data(crawler_filepaths_raw)
                
                if parsed_filepaths:
                    st.write(f"Found {len(parsed_filepaths)} files:")
                    images_found = 0
                    for i, filepath in enumerate(parsed_filepaths):
                        st.write(f"**File {i+1}**: {filepath}")
                        if isinstance(filepath, str) and is_image_file(filepath):
                            load_success = try_load_image_with_blip_caption(filepath, crawler_bucket, None, "No caption available")
                            if load_success:
                                images_loaded = True
                                images_found += 1
                                if images_found >= 3:  # Limit to first 3 images
                                    break
                
            # Priority 4: crawler_artifact_filepath
            if not images_loaded and 'crawler_artifact_filepath' in image_sources:
                st.markdown("#### 📄 Crawler Artifact File")
                artifact_filepath = image_sources['crawler_artifact_filepath']
                st.write(f"**Artifact filepath**: {artifact_filepath}")
                
                if is_image_file(artifact_filepath):
                    load_success = try_load_image_with_blip_caption(artifact_filepath, crawler_bucket, None, "No caption available")
                    if load_success:
                        images_loaded = True
                else:
                    st.write("Artifact filepath doesn't appear to be an image file")
            
            # Priority 5: Search in text_content
            if not images_loaded and 'text_content' in image_sources:
                st.markdown("#### 📝 Text Content Image Search")
                text_content_raw = image_sources['text_content']
                
                # Try to extract image references from text_content
                try:
                    if isinstance(text_content_raw, str):
                        text_content = json.loads(text_content_raw)
                    else:
                        text_content = text_content_raw
                    
                    images_in_content = extract_images_from_content(text_content)
                    if images_in_content:
                        st.write(f"Found {len(images_in_content)} potential images in text content:")
                        for img_ref in images_in_content:
                            st.write(f"- {img_ref}")
                            load_success = try_load_image_with_blip_caption(img_ref, crawler_bucket, None, "No caption available")
                            if load_success:
                                images_loaded = True
                                break
                    else:
                        st.write("No image references found in text_content")
                except Exception as e:
                    st.write(f"Error parsing text_content: {str(e)}")
            
            # Priority 6: Search in content_metadata
            if not images_loaded and 'content_metadata' in image_sources:
                st.markdown("#### 🗃️ Content Metadata Image Search")
                content_metadata_raw = image_sources['content_metadata']
                
                try:
                    if isinstance(content_metadata_raw, str):
                        content_metadata = json.loads(content_metadata_raw)
                    else:
                        content_metadata = content_metadata_raw
                    
                    if isinstance(content_metadata, dict):
                        for key, value in content_metadata.items():
                            if isinstance(value, str) and is_image_file(value):
                                st.write(f"Found image in metadata field '{key}': {value}")
                                load_success = try_load_image_with_blip_caption(value, crawler_bucket, None, "No caption available")
                                if load_success:
                                    images_loaded = True
                                    break
                except Exception as e:
                    st.write(f"Error parsing content_metadata: {str(e)}")
            
            # Final fallback
            if not images_loaded:
                st.error("❌ No images could be loaded from any source")
                st.write("**Debugging suggestions:**")
                st.write("1. Check if the crawler_output_bucket is correct")
                st.write("2. Verify the file paths exist in the bucket")
                st.write("3. Check if the artifact is actually a video (some may be text-only)")
                st.write("4. Look at the raw data above to understand the structure")
                st.write("5. BLIP Large model may not be loaded - check model initialization")
            else:
                st.success("✅ Successfully loaded images with BLIP Large caption analysis!")
                
        except Exception as e:
            st.error(f"Error in comprehensive image search: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)

    with main_col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Video Content")
        try:
            # Display text content from metadata table
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
                    text_content = text_content.tolist() if hasattr(text_content_raw, 'tolist') else list(text_content_raw)
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
            
            # Display additional metadata from content_metadata field
            content_metadata_raw = current_item.get('content_metadata')
            if content_metadata_raw:
                try:
                    if isinstance(content_metadata_raw, str):
                        content_metadata = json.loads(content_metadata_raw)
                    else:
                        content_metadata = content_metadata_raw
                    
                    if isinstance(content_metadata, dict) and content_metadata:
                        st.markdown("**Additional Content Metadata:**")
                        for key, value in content_metadata.items():
                            if value and key not in ['text_content']:  # Avoid duplicating text_content
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        st.markdown("---")
                except Exception as e:
                    st.write(f"Could not parse content_metadata: {str(e)}")
            
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

    # Caption Audit Section - Enhanced with BLIP captions
    st.markdown("### 📝 Caption Audit")
    
    # Get current frame caption for audit and check if BLIP was used
    current_frame_caption = "No caption available"
    blip_generated_caption = None
    
    asset_classifications_raw = current_item.get('asset_classifications', '[]')
    
    # Parse asset classifications for caption
    asset_classifications = []
    if isinstance(asset_classifications_raw, str):
        try:
            asset_classifications = json.loads(asset_classifications_raw)
        except:
            pass
    elif isinstance(asset_classifications_raw, list):
        asset_classifications = asset_classifications_raw
    elif hasattr(asset_classifications_raw, '__iter__') and not isinstance(asset_classifications_raw, str):
        try:
            asset_classifications = asset_classifications_raw.tolist() if hasattr(asset_classifications_raw, 'tolist') else list(asset_classifications_raw)
        except:
            pass
    
    if asset_classifications and len(asset_classifications) > 0:
        image_classifications = [c for c in asset_classifications if isinstance(c, dict) and c.get('filepath')]
        if image_classifications and current_image_index < len(image_classifications):
            current_image_classification = image_classifications[current_image_index]
            current_frame_caption = current_image_classification.get('caption', 'No caption available')
    
    # Show both original and BLIP captions if available
    caption_col1, caption_col2 = st.columns([2, 1])
    
    with caption_col1:
        st.markdown("**Current Frame Caption:**")
        st.text_area(
            "Original caption:", 
            value=current_frame_caption, 
            height=80, 
            disabled=True,
            key=f"{page_key}caption_display"
        )
        
        # Show BLIP caption area (will be populated if BLIP was used above)
        if 'blip_caption_session' not in st.session_state:
            st.session_state.blip_caption_session = {}
        
        current_record_key = f"{page_key}_{current_index}_{current_image_index}"
        if current_record_key in st.session_state.blip_caption_session:
            st.markdown("**BLIP Large Generated Caption:**")
            st.text_area(
                "BLIP Large caption:", 
                value=st.session_state.blip_caption_session[current_record_key], 
                height=80, 
                disabled=True,
                key=f"{page_key}blip_caption_display"
            )
            blip_generated_caption = st.session_state.blip_caption_session[current_record_key]
    
    with caption_col2:
        st.markdown("**Caption Audit Questions:**")
        
        # Caption source selection
        caption_source_options = ["Original Caption"]
        if blip_generated_caption:
            caption_source_options.append("BLIP Large Generated")
            caption_source_options.append("Both Combined")
        
        caption_source = st.radio(
            "Which caption are you evaluating?",
            caption_source_options,
            index=0,
            key=f"{page_key}caption_source"
        )
        
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
                    # Caption audit data - Enhanced with BLIP info
                    'current_frame_caption': current_frame_caption,
                    'blip_generated_caption': blip_generated_caption if blip_generated_caption else 'None',
                    'caption_source_evaluated': caption_source,
                    'caption_accuracy': caption_accuracy,
                    'caption_completeness': caption_completeness,
                    'caption_relevance': caption_relevance,
                    'caption_issues': ', '.join(caption_issues) if caption_issues else 'None',
                    'improved_caption': improved_caption.strip() if improved_caption.strip() else 'None',
                    'caption_quality_score': caption_quality_score,
                    'frame_number': current_image_index + 1,
                    # Additional metadata table info
                    'crawler_output_bucket': current_item.get('crawler_output_bucket', 'N/A'),
                    'screenshot_filepath': current_item.get('screenshot_filepath', 'N/A'),
                    'metadata_created_at': str(current_item.get('metadata_created_at', 'N/A')),
                }
                st.session_state.corrections.append(correction)
                st.success("Meta video correction added with enhanced metadata!")
                
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
                    st.success(f"Saved {len(st.session_state.corrections)} Meta video corrections with enhanced metadata!")
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