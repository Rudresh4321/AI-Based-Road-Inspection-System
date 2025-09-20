# core_functions.py
"""
Core functionality for FixMyStreet AI Road Inspection System
Contains shared functions for detection, database operations, and utilities
"""

import os
import io
import time
import hashlib
import tempfile
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill

from geopy.geocoders import Nominatim
from ultralytics import YOLO

import streamlit as st
from PIL import Image

# -------------------------
# Constants and Configuration
# -------------------------
USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin",
        "name": "Municipal Administrator"
    },
    "user": {
        "password": "user123",
        "role": "inspector",
        "name": "Road Inspector"
    }
}

# Database files
ACTIVE_REPAIRS_FILE = "road_inspection_database.xlsx"
FIXED_REPAIRS_FILE = "fixed_repairs_database.xlsx"

# Status options
STATUS_OPTIONS = ["Reported", "In Progress", "Fixed", "Verified", "Cancelled"]
SEVERITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
DEFECT_TYPES = ["pothole", "crack", "alligator_crack", "longitudinal_crack", "transverse_crack", "block_crack", "joint_crack", "other"]
SIZE_OPTIONS = ["small", "medium", "large", "extra_large"]

# -------------------------
# Utility Functions
# -------------------------
def hash_password(password: str) -> str:
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def safe_button(label: str, **kwargs):
    """Safe wrapper for st.button with compatibility across Streamlit versions"""
    try:
        return st.button(label, **kwargs)
    except TypeError:
        # Remove problematic kwargs for older versions
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['use_container_width', 'type', 'button_type']}
        try:
            return st.button(label, **safe_kwargs)
        except TypeError:
            return st.button(label)

def safe_rerun():
    """Safe rerun function compatible with different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page manually")

def show_success_message(message: str):
    """Show success message that persists across reruns"""
    st.session_state.show_success = True
    st.session_state.success_message = message

def display_session_messages():
    """Display any pending session messages"""
    if st.session_state.show_success:
        st.success(st.session_state.success_message)
        st.session_state.show_success = False
        st.session_state.success_message = ""

# -------------------------
# Model Loading
# -------------------------
@st.cache(show_spinner=False, allow_output_mutation=True)
def load_yolo_model():
    """Load YOLO model with improved caching and error handling"""
    try:
        candidate_paths = [
            'best.pt',
            './best.pt',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt'),
            '../best.pt'
        ]
        
        model_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            return None, "⚠️ YOLO model file 'best.pt' not found. Detection features will be unavailable."
            
        model = YOLO(model_path)
        return model, f"✅ YOLO model loaded successfully from: {model_path}"
        
    except Exception as e:
        return None, f"❌ Error loading YOLO model: {str(e)}"

# -------------------------
# Geocoder Initialization
# -------------------------
def initialize_geocoder():
    """Initialize geocoding service"""
    try:
        geolocator = Nominatim(user_agent="fixmystreet_road_inspection_v2")
        return geolocator
    except Exception as e:
        st.warning(f"Geocoding service unavailable: {e}")
        return None

# -------------------------
# Repair Method Functions
# -------------------------
def get_repair_method(defect_type: str, severity: str) -> str:
    """Get appropriate repair method based on defect type and severity"""
    repair_methods = {
        'pothole': {
            'Low': 'Cold Mix Patching',
            'Medium': 'Hot Mix Asphalt Patching',
            'High': 'Full Depth Reconstruction',
            'Critical': 'Complete Road Section Rebuild'
        },
        'crack': {
            'Low': 'Crack Sealing',
            'Medium': 'Crack Routing and Sealing',
            'High': 'Surface Treatment + Overlay',
            'Critical': 'Full Depth Repair'
        },
        'alligator_crack': {
            'Low': 'Surface Sealing',
            'Medium': 'Milling and Thin Overlay',
            'High': 'Deep Milling and Thick Overlay',
            'Critical': 'Full Depth Reconstruction'
        }
    }
    
    default_methods = {
        'Low': 'Basic Surface Repair',
        'Medium': 'Standard Repair Method',
        'High': 'Major Reconstruction',
        'Critical': 'Complete Replacement'
    }
    
    return repair_methods.get(defect_type.lower(), default_methods).get(severity, 'Standard Repair')

# -------------------------
# Database Management Functions
# -------------------------
def get_database_headers():
    """Get standardized database headers"""
    return [
        'Detection_ID', 'Timestamp', 'Location', 'Road_Name',
        'GPS_Lat', 'GPS_Lon', 'Defect_Type', 'Confidence',
        'Severity', 'Size_Category', 'Repair_Method', 'Fix_Type',
        'Notes', 'Inspector', 'Status', 'Priority_Score', 'Date_Fixed'
    ]

def initialize_database_file(file_path: str, sheet_name: str = "Road_Inspection_Data") -> bool:
    """Initialize Excel database file with proper formatting"""
    try:
        if not os.path.exists(file_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = sheet_name
            
            headers = get_database_headers()
            
            # Create header row with formatting
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="2c3e50", end_color="2c3e50", fill_type="solid")
            
            # Auto-adjust column widths
            for col in range(1, len(headers) + 1):
                ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = 15
            
            wb.save(file_path)
            return True
        return True
    except Exception as e:
        st.error(f"❌ Error initializing database file {file_path}: {str(e)}")
        return False

def calculate_priority_score(severity: str, defect_type: str, size_category: str) -> int:
    """Calculate priority score for repair scheduling"""
    severity_scores = {'Low': 1, 'Medium': 3, 'High': 7, 'Critical': 10}
    defect_scores = {'crack': 1, 'pothole': 3, 'alligator_crack': 5}
    size_scores = {'small': 1, 'medium': 2, 'large': 4, 'extra_large': 6}
    
    return (severity_scores.get(severity, 3) * 3 + 
            defect_scores.get(defect_type.lower(), 2) * 2 + 
            size_scores.get(size_category, 2))

def save_detection(detection_data: dict, file_path: str = ACTIVE_REPAIRS_FILE) -> bool:
    """Save detection data to specified database file"""
    try:
        initialize_database_file(file_path)
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        
        # Calculate priority score
        priority_score = calculate_priority_score(
            detection_data.get('Severity', 'Medium'),
            detection_data.get('Defect_Type', 'other'),
            detection_data.get('Size_Category', 'medium')
        )
        
        next_row = ws.max_row + 1
        row_data = [
            detection_data.get('Detection_ID'),
            detection_data.get('Timestamp'),
            detection_data.get('Location'),
            detection_data.get('Road_Name'),
            detection_data.get('GPS_Lat'),
            detection_data.get('GPS_Lon'),
            detection_data.get('Defect_Type'),
            detection_data.get('Confidence'),
            detection_data.get('Severity'),
            detection_data.get('Size_Category'),
            detection_data.get('Repair_Method'),
            detection_data.get('Fix_Type'),
            detection_data.get('Notes'),
            detection_data.get('Inspector'),
            'Reported',  # Default status
            priority_score,
            detection_data.get('Date_Fixed', '')
        ]
        
        for col, value in enumerate(row_data, 1):
            ws.cell(row=next_row, column=col, value=value)
        
        wb.save(file_path)
        return True
        
    except Exception as e:
        st.error(f"❌ Error saving detection data: {str(e)}")
        return False

def apply_header_formatting(worksheet):
    """Apply consistent header formatting to worksheet"""
    try:
        for col in range(1, worksheet.max_column + 1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="2c3e50", end_color="2c3e50", fill_type="solid")
    except Exception:
        pass

# -------------------------
# YOLO Detection Processing
# -------------------------
def extract_box_information(box, model_obj):
    """Safely extract class name and confidence from YOLO detection box"""
    try:
        # Handle different YOLO versions and box formats
        cls_val = None
        conf_val = None
        
        # Try different attribute access patterns
        if hasattr(box, 'cls'):
            cls_val = box.cls
        elif hasattr(box, 'class_id'):
            cls_val = box.class_id
            
        if hasattr(box, 'conf'):
            conf_val = box.conf
        elif hasattr(box, 'confidence'):
            conf_val = box.confidence
        
        # Convert to Python primitives
        try:
            if hasattr(cls_val, '__len__') and len(cls_val) > 0:
                cls_int = int(cls_val[0])
            else:
                cls_int = int(cls_val) if cls_val is not None else 0
        except (TypeError, ValueError):
            cls_int = 0
            
        try:
            if hasattr(conf_val, '__len__') and len(conf_val) > 0:
                conf_float = float(conf_val[0])
            else:
                conf_float = float(conf_val) if conf_val is not None else 0.0
        except (TypeError, ValueError):
            conf_float = 0.0
        
        # Get class name
        class_name = f"defect_{cls_int}"
        if model_obj and hasattr(model_obj, 'names'):
            try:
                if isinstance(model_obj.names, dict):
                    class_name = model_obj.names.get(cls_int, f"defect_{cls_int}")
                elif isinstance(model_obj.names, list) and cls_int < len(model_obj.names):
                    class_name = model_obj.names[cls_int]
            except Exception:
                pass
                
        return class_name, conf_float
        
    except Exception as e:
        st.warning(f"Error extracting box information: {e}")
        return "unknown_defect", 0.0

def process_image_detections(prediction, location: str, road_name: str, lat: float, lon: float, 
                           severity: str, size_category: str, notes: str, model_obj) -> int:
    """Process YOLO detections from image and save to database"""
    if not model_obj or not prediction:
        return 0
        
    saved_count = 0
    try:
        if len(prediction) > 0 and prediction[0].boxes is not None:
            boxes = prediction[0].boxes
            
            for i, box in enumerate(boxes):
                class_name, confidence = extract_box_information(box, model_obj)
                repair_method = get_repair_method(class_name, severity)
                
                detection_data = {
                    'Detection_ID': f"DET_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}",
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Location': location,
                    'Road_Name': road_name,
                    'GPS_Lat': round(float(lat), 6),
                    'GPS_Lon': round(float(lon), 6),
                    'Defect_Type': class_name,
                    'Confidence': round(confidence, 3),
                    'Severity': severity,
                    'Size_Category': size_category,
                    'Repair_Method': repair_method,
                    'Fix_Type': 'To Be Decided',  # Default fix type
                    'Notes': notes,
                    'Inspector': st.session_state.user_name
                }
                
                if save_detection(detection_data):
                    saved_count += 1
                    
    except Exception as e:
        st.error(f"❌ Error processing image detections: {str(e)}")
        
    return saved_count

def process_video_detections(video_path: str, location: str, road_name: str, lat: float, lon: float,
                           severity: str, size_category: str, notes: str, model_obj) -> tuple:
    """Process YOLO detections from video and save to database"""
    if not model_obj:
        return None, 0
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Error opening video file")
        return None, 0
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    output_path = f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    total_saved = 0
    frame_count = 0
    
    try:
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Progress tracking
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0.0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f'Processing frame {frame_count}/{total_frames}')
            
            try:
                # Run detection
                prediction = model_obj.predict(frame, verbose=False)
                
                # Save detections every 30 frames (1 second at 30 fps)
                if frame_count % 30 == 0:
                    saved_count = process_image_detections(
                        prediction, location, road_name, lat, lon, severity, size_category, notes, model_obj
                    )
                    total_saved += saved_count
                
                # Create annotated frame
                if prediction and len(prediction) > 0:
                    annotated_frame = prediction[0].plot()
                    # Convert RGB to BGR for video writer
                    if annotated_frame is not None:
                        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    else:
                        out.write(frame)
                else:
                    out.write(frame)
                    
            except Exception as e:
                st.warning(f"Error processing frame {frame_count}: {str(e)}")
                out.write(frame)  # Write original frame on error
                continue
        
        # Cleanup
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        return output_path, total_saved
        
    except Exception as e:
        cap.release()
        if 'out' in locals():
            out.release()
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Error processing video: {e}")

# -------------------------
# Authentication Functions
# -------------------------
def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user and return user info"""
    if username in USERS and USERS[username]["password"] == password:
        return USERS[username]
    return None

def login_interface():
    """Login interface for all users (no nested columns)"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #2c3e50; margin-bottom: 0.5rem;'>🛣️ FixMyStreet</h1>
        <h3 style='color: #7f8c8d; font-weight: 300;'>AI Road Inspection System</h3>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 Please Login to Continue")
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        login_clicked = safe_button("🚀 Login")
        demo_clicked = safe_button("ℹ️ Demo Info")
        if login_clicked:
            user_info = authenticate_user(username, password)
            if user_info:
                st.session_state.logged_in = True
                st.session_state.user_role = user_info["role"]
                st.session_state.user_name = user_info["name"]
                st.session_state.username = username
                show_success_message(f"Welcome, {st.session_state.user_name}!")
                safe_rerun()
            else:
                st.error("❌ Invalid username or password")
        if demo_clicked:
            st.info("""
            **Demo Credentials:**
            **👨‍💼 Admin Access:**
            - Username: `admin`
            - Password: `admin123`
            - Features: Full database access, record management, reports
            **👷‍♂️ Inspector Access:**
            - Username: `user`
            - Password: `user123`
            - Features: Defect detection only
            """)

def logout():
    """Logout function"""
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.session_state.username = None
    show_success_message("👋 Successfully logged out!")
    safe_rerun()

# -------------------------
# Detection Interface (Shared)
# -------------------------
def detection_interface(model, geolocator):
    """Shared detection interface for both user types"""
    st.header("🔍 AI Road Defect Detection")
    
    if not model:
        st.error("❌ YOLO model not available. Please ensure 'best.pt' is in the application directory.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        # Location information
        st.subheader("📍 Location Information")
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input("📍 Location Description", placeholder="e.g., MG Road, Bangalore")
            road_name = st.text_input("🛣️ Road Name", placeholder="e.g., MG Road")
            # Removed Auto-detect Location from GPS feature
        with col2:
            lat = st.number_input("🌍 Latitude", value=12.9716, format="%.6f")
            lon = st.number_input("🌏 Longitude", value=77.5946, format="%.6f")
        
        # Additional metadata
        st.subheader("ℹ️ Additional Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity = st.selectbox("⚠️ Severity Level", SEVERITY_OPTIONS, index=1)
        
        with col2:
            size_category = st.selectbox("📏 Size Category", SIZE_OPTIONS, index=1)
        
        with col3:
            notes = st.text_area("📝 Additional Notes", placeholder="Optional notes about the defect...")
        
        # Process button
        if safe_button("🚀 Process Detection", use_container_width=True, type="primary"):
            if not location or not road_name:
                st.error("❌ Please provide location and road name")
                return
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                file_type = uploaded_file.type
                
                if file_type.startswith('image'):
                    # Process image
                    st.info("🔄 Processing image...")
                    
                    # Load and display original image
                    image = Image.open(temp_path)
                    st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Run detection
                    prediction = model.predict(temp_path, verbose=False)
                    
                    # Display results
                    if prediction and len(prediction) > 0:
                        annotated_image = prediction[0].plot()
                        st.image(annotated_image, caption="Detection Results", use_column_width=True)
                        
                        # Save detections
                        saved_count = process_image_detections(
                            prediction, location, road_name, lat, lon, severity, size_category, notes, model
                        )
                        
                        if saved_count > 0:
                            st.success(f"✅ Successfully saved {saved_count} detection(s) to database!")
                        else:
                            st.warning("⚠️ No defects detected in the image")
                    else:
                        st.warning("⚠️ No defects detected in the image")
                
                elif file_type.startswith('video'):
                    # Process video
                    st.info("🔄 Processing video... This may take a while.")
                    
                    try:
                        output_path, saved_count = process_video_detections(
                            temp_path, location, road_name, lat, lon, severity, size_category, notes, model
                        )
                        
                        if output_path and os.path.exists(output_path):
                            st.success(f"✅ Video processed successfully! Saved {saved_count} detections.")
                            
                            # Offer download of processed video
                            with open(output_path, 'rb') as video_file:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=video_file.read(),
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                            
                            # Cleanup
                            os.unlink(output_path)
                        else:
                            st.error("❌ Error processing video")
                            
                    except Exception as e:
                        st.error(f"❌ Video processing failed: {str(e)}")
                
                # Cleanup temp file
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)

def display_database_statistics(df: pd.DataFrame, title: str):
    """Display database statistics in a nice format (robust cost handling)"""
    st.subheader(f"📊 {title}")
    if df.empty:
        st.info(f"No records found in {title.lower()}")
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if 'Severity' in df.columns:
            high_priority = len(df[df['Severity'].isin(['High', 'Critical'])])
            st.metric("High/Critical Priority", high_priority)
        else:
            st.metric("High Priority", "N/A")
    with col3:
        if 'Status' in df.columns:
            pending = len(df[df['Status'].isin(['Reported', 'In Progress'])])
            st.metric("Pending Repairs", pending)
        else:
            st.metric("Pending", "N/A")
    with col4:
        if 'Fix_Type' in df.columns:
            # Count fix types
            fix_type_counts = df['Fix_Type'].value_counts()
            for fix_type, count in fix_type_counts.items():
                st.metric(f"{fix_type} Repairs", count)
        else:
            st.metric("Fix Type Stats", "N/A")