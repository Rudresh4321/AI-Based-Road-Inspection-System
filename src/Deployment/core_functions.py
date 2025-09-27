# core_functions.py
"""
Core functionality for FixMyStreet AI Road Inspection System
Contains shared functions for detection, database operations, and utilities
Updated with SQLite integration to replace Excel files
"""

import os
import io
import time
import hashlib
import tempfile
import sqlite3
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
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------------
# Constants and Configuration
# -------------------------
# SQLite Database file (replaces all Excel files)
DB_FILE = "road_inspection.db"

# Legacy file paths (kept for backward compatibility during transition)
ACTIVE_REPAIRS_FILE = "road_inspection_database.xlsx"
FIXED_REPAIRS_FILE = "fixed_repairs_database.xlsx"
USERS_FILE = 'users.xlsx'

# Status options
STATUS_OPTIONS = ["Reported", "In Progress", "Fixed", "Verified", "Cancelled"]
SEVERITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
DEFECT_TYPES = ["pothole", "crack", "alligator_crack", "longitudinal_crack", "transverse_crack", "block_crack", "joint_crack", "other"]
SIZE_OPTIONS = ["small", "medium", "large", "extra_large"]

# -------------------------
# SQLite Database Management
# -------------------------
def get_db_connection():
    """Get SQLite database connection"""
    return sqlite3.connect(DB_FILE)

def initialize_database():
    """Initialize SQLite database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table (replaces users.xlsx)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'inspector',
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create repairs table (replaces all Excel repair files)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT UNIQUE NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location TEXT NOT NULL,
            road_name TEXT,
            gps_lat REAL,
            gps_lon REAL,
            defect_type TEXT NOT NULL,
            confidence REAL,
            severity TEXT NOT NULL,
            size_category TEXT,
            repair_method TEXT,
            fix_type TEXT DEFAULT 'To Be Decided',
            notes TEXT,
            inspector TEXT,
            status TEXT DEFAULT 'Reported',
            priority_score INTEGER,
            date_fixed TEXT,
            assigned_to TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create OTP table for email verification
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS otp_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_used BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Create default admin user if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        hashed_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute(
            "INSERT INTO users (username, password, role, name) VALUES (?, ?, ?, ?)",
            ('admin', hashed_password, 'admin', 'Administrator')
        )
    
    conn.commit()
    conn.close()

# -------------------------
# Utility Functions
# -------------------------
def hash_password(password: str) -> str:
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

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
    if st.session_state.get('show_success', False):
        st.success(st.session_state.success_message)
        st.session_state.show_success = False
        st.session_state.success_message = ""

def logout():
    """Logout function - clear session state"""
    # Clear all session state keys on logout for a clean slate
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    show_success_message("Successfully logged out!")
    safe_rerun()

# -------------------------
# OTP and Email Functions
# -------------------------
def generate_otp(length=6):
    """Generate a secure random OTP."""
    return "".join([str(secrets.randbelow(10)) for _ in range(length)])

def send_otp_email(recipient_email: str, otp: str):
    """Send an OTP to the user's email address using st.secrets."""
    try:
        # Get sender credentials from Streamlit secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"] # App Password

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your OTP for FixMyStreet Registration"
        message["From"] = sender_email
        message["To"] = recipient_email

        text = f"Hi,\n\nYour One-Time Password (OTP) for registering on FixMyStreet is: {otp}\n\nThis OTP is valid for 5 minutes.\n\nThank you!"
        html = f"""
        <html>
        <body>
            <p>Hi,</p>
            <p>Your One-Time Password (OTP) for registering on FixMyStreet is: <b>{otp}</b></p>
            <p>This OTP is valid for 5 minutes.</p>
            <p>Thank you!</p>
        </body>
        </html>
        """

        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True
    except Exception as e:
        st.error(f"Failed to send OTP email: {e}")
        st.error("Please ensure email secrets are configured correctly by the administrator.")
        return False

def store_otp(email: str, otp: str):
    """Store OTP in SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clean up old OTPs for this email
        cursor.execute("DELETE FROM otp_tokens WHERE email = ?", (email,))
        
        # Insert new OTP
        cursor.execute(
            "INSERT INTO otp_tokens (email, otp) VALUES (?, ?)",
            (email, otp)
        )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error storing OTP: {str(e)}")
        return False

def verify_otp(email: str, otp: str) -> bool:
    """Verify OTP from SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM otp_tokens 
            WHERE email = ? AND otp = ? AND is_used = FALSE 
            AND datetime(created_at, '+5 minutes') > datetime('now')
        """, (email, otp))
        
        result = cursor.fetchone()
        
        if result:
            # Mark OTP as used
            cursor.execute(
                "UPDATE otp_tokens SET is_used = TRUE WHERE id = ?",
                (result[0],)
            )
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
        
    except Exception as e:
        st.error(f"Error verifying OTP: {str(e)}")
        return False

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource(show_spinner=False)
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
            return None, "YOLO model file 'best.pt' not found. Detection features will be unavailable."
            
        model = YOLO(model_path)
        return model, f"YOLO model loaded successfully from: {model_path}"
        
    except Exception as e:
        return None, f"Error loading YOLO model: {str(e)}"

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

def calculate_priority_score(severity: str, defect_type: str, size_category: str) -> int:
    """Calculate priority score for repair scheduling"""
    severity_scores = {'Low': 1, 'Medium': 3, 'High': 7, 'Critical': 10}
    defect_scores = {'crack': 1, 'pothole': 3, 'alligator_crack': 5}
    size_scores = {'small': 1, 'medium': 2, 'large': 4, 'extra_large': 6}
    
    return (severity_scores.get(severity, 3) * 3 + 
            defect_scores.get(defect_type.lower(), 2) * 2 + 
            size_scores.get(size_category, 2))

# -------------------------
# Database Operations (SQLite-based)
# -------------------------
def save_detection(detection_data: dict) -> bool:
    """Save detection data to SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate priority score
        priority_score = calculate_priority_score(
            detection_data.get('Severity', 'Medium'),
            detection_data.get('Defect_Type', 'other'),
            detection_data.get('Size_Category', 'medium')
        )
        
        cursor.execute('''
            INSERT INTO repairs (
                detection_id, timestamp, location, road_name, gps_lat, gps_lon,
                defect_type, confidence, severity, size_category, repair_method,
                fix_type, notes, inspector, status, priority_score, date_fixed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
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
            detection_data.get('Fix_Type', 'To Be Decided'),
            detection_data.get('Notes'),
            detection_data.get('Inspector'),
            'Reported',  # Default status
            priority_score,
            detection_data.get('Date_Fixed', '')
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error saving detection data: {str(e)}")
        return False

def get_repairs_data(status_filter=None, inspector_filter=None):
    """Get repairs data from SQLite database"""
    try:
        conn = get_db_connection()
        
        query = "SELECT * FROM repairs"
        params = []
        conditions = []
        
        if status_filter:
            if isinstance(status_filter, list):
                placeholders = ','.join(['?' for _ in status_filter])
                conditions.append(f"status IN ({placeholders})")
                params.extend(status_filter)
            else:
                conditions.append("status = ?")
                params.append(status_filter)
        
        if inspector_filter:
            conditions.append("inspector = ?")
            params.append(inspector_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
        
    except Exception as e:
        st.error(f"Error retrieving repairs data: {str(e)}")
        return pd.DataFrame()

def update_repair_status(detection_id: str, new_status: str, assigned_to: str = None):
    """Update repair status in SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if assigned_to:
            cursor.execute(
                "UPDATE repairs SET status = ?, assigned_to = ?, updated_at = ? WHERE detection_id = ?",
                (new_status, assigned_to, datetime.now().isoformat(), detection_id)
            )
        else:
            cursor.execute(
                "UPDATE repairs SET status = ?, updated_at = ? WHERE detection_id = ?",
                (new_status, datetime.now().isoformat(), detection_id)
            )
        
        # If status is Fixed or Verified, set date_fixed
        if new_status in ['Fixed', 'Verified']:
            cursor.execute(
                "UPDATE repairs SET date_fixed = ? WHERE detection_id = ?",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), detection_id)
            )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error updating repair status: {str(e)}")
        return False

def delete_repair(detection_id: str):
    """Delete repair from SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM repairs WHERE detection_id = ?", (detection_id,))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error deleting repair: {str(e)}")
        return False

def get_repair_statistics():
    """Get repair statistics from SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get status counts
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM repairs 
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Get severity counts
        cursor.execute("""
            SELECT severity, COUNT(*) as count 
            FROM repairs 
            GROUP BY severity
        """)
        severity_counts = dict(cursor.fetchall())
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM repairs")
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total_count,
            'status_counts': status_counts,
            'severity_counts': severity_counts,
            'active': status_counts.get('Reported', 0) + status_counts.get('In Progress', 0),
            'completed': status_counts.get('Fixed', 0) + status_counts.get('Verified', 0)
        }
        
    except Exception as e:
        st.error(f"Error getting repair statistics: {str(e)}")
        return {'total': 0, 'status_counts': {}, 'severity_counts': {}, 'active': 0, 'completed': 0}

# -------------------------
# User Authentication (SQLite-based)
# -------------------------
def authenticate_user(username, password):
    """Authenticate user against SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute(
            "SELECT id, username, role, name, email FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2],
                'name': result[3],
                'email': result[4]
            }
        return None
        
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None

def register_user(username, password, name):
    """Register new user in SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            st.error("Username (Email) already exists.")
            conn.close()
            return False
        
        # Insert new user
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password, role, name, email) VALUES (?, ?, ?, ?, ?)",
            (username, hashed_password, 'inspector', name, username)
        )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

# -------------------------
# Login Interface (Shared)
# -------------------------
def login_interface():
    """Shared login and registration interface with OTP verification"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #2c3e50; margin-bottom: 0.5rem;'>🛣️ FixMyStreet</h1>
        <h3 style='color: #7f8c8d; font-weight: 300;'>AI Road Inspection System</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username or Email", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_role = user['role']
                        st.session_state.user_name = user['name']
                        st.session_state.username = username
                        st.session_state.user_id = user['id']
                        show_success_message(f"Welcome, {user['name']}!")
                        safe_rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            with st.form("register_form"):
                st.markdown("##### Step 1: Enter Your Details")
                new_email = st.text_input("Gmail Address", key="reg_email", help="You will receive an OTP at this address.")
                new_name = st.text_input("Full Name", key="reg_name")
                new_password = st.text_input("Choose a Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")

                st.markdown("---")
                st.markdown("##### Step 2: Verify Your Email")
                
                col_otp_btn, col_otp_field = st.columns([1, 2])

                with col_otp_btn:
                    if st.form_submit_button("✉️ Send OTP"):
                        if not new_email.endswith('@gmail.com'):
                            st.error("Please enter a valid Gmail address.")
                        else:
                            otp = generate_otp()
                            if send_otp_email(new_email, otp):
                                if store_otp(new_email, otp):
                                    st.session_state['otp_sent'] = True
                                    st.session_state['otp_email'] = new_email
                                    st.session_state['otp_time'] = datetime.now()
                                    st.info("OTP has been sent to your email. Please check your inbox.")
                                else:
                                    st.error("Failed to store OTP. Please try again.")

                with col_otp_field:
                    otp_input = st.text_input("Enter OTP", key="reg_otp", max_chars=6)

                st.markdown("---")
                st.markdown("##### Step 3: Complete Registration")
                register_submitted = st.form_submit_button("✅ Register Account")

                if register_submitted:
                    # All validations happen on final submission
                    otp_sent = st.session_state.get('otp_sent', False)
                    otp_email_in_session = st.session_state.get('otp_email')
                    otp_time_in_session = st.session_state.get('otp_time')

                    if new_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif not all([new_email, new_name, new_password, otp_input]):
                        st.error("All fields, including OTP, are required!")
                    elif not otp_sent or otp_email_in_session != new_email:
                        st.error("Please send and verify OTP for the entered email address first.")
                    elif otp_time_in_session and (datetime.now() - otp_time_in_session) > timedelta(minutes=5):
                        st.error("OTP has expired. Please request a new one.")
                    elif not verify_otp(new_email, otp_input):
                        st.error("Invalid or expired OTP. Please try again.")
                    else:
                        # All checks passed, proceed with registration
                        if register_user(new_email, new_password, new_name):
                            st.success("Registration successful! Please log in.")
                            # Clear OTP from session state after successful use
                            st.session_state.pop('otp_sent', None)
                            st.session_state.pop('otp_email', None)
                            st.session_state.pop('otp_time', None)

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
                    'Fix_Type': 'To Be Decided',
                    'Notes': notes,
                    'Inspector': st.session_state.user_name
                }
                
                if save_detection(detection_data):
                    saved_count += 1
                    
    except Exception as e:
        st.error(f"Error processing image detections: {str(e)}")
        
    return saved_count

def process_video_detections(video_path: str, location: str, road_name: str, lat: float, lon: float,
                           severity: str, size_category: str, notes: str, model_obj) -> tuple:
    """Process YOLO detections from video and save to database"""
    if not model_obj:
        return None, 0
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
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
# Detection Interface (Shared)
# -------------------------
def detection_interface(model, geolocator):
    """Shared detection interface for both user types"""
    st.header("AI Road Defect Detection")
    
    if not model:
        st.error("YOLO model not available. Please ensure 'best.pt' is in the application directory.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        # Location information
        st.subheader("Location Information")
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input("Location Description", placeholder="e.g., MG Road, Bangalore")
            road_name = st.text_input("Road Name", placeholder="e.g., MG Road")
        with col2:
            lat = st.number_input("Latitude", value=12.9716, format="%.6f")
            lon = st.number_input("Longitude", value=77.5946, format="%.6f")
        
        # Additional metadata
        st.subheader("Additional Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity = st.selectbox("Severity Level", SEVERITY_OPTIONS, index=1)
        
        with col2:
            size_category = st.selectbox("Size Category", SIZE_OPTIONS, index=1)
        
        with col3:
            notes = st.text_area("Additional Notes", placeholder="Optional notes about the defect...")
        
        # Process button
        if st.button("Process Detection", use_container_width=True, type="primary"):
            if not location or not road_name:
                st.error("Please provide location and road name")
                return
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                file_type = uploaded_file.type
                
                if file_type.startswith('image'):
                    # Process image
                    st.info("Processing image...")
                    
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
                            st.success(f"Successfully saved {saved_count} detection(s) to database!")
                        else:
                            st.warning("No defects detected in the image")
                    else:
                        st.warning("No defects detected in the image")
                
                elif file_type.startswith('video'):
                    # Process video
                    st.info("Processing video... This may take a while.")
                    
                    try:
                        output_path, saved_count = process_video_detections(
                            temp_path, location, road_name, lat, lon, severity, size_category, notes, model
                        )
                        
                        if output_path and os.path.exists(output_path):
                            st.success(f"Video processed successfully! Saved {saved_count} detections.")
                            
                            # Offer download of processed video
                            with open(output_path, 'rb') as video_file:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=video_file.read(),
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                            
                            # Cleanup
                            os.unlink(output_path)
                        else:
                            st.error("Error processing video")
                            
                    except Exception as e:
                        st.error(f"Video processing failed: {str(e)}")
                
                # Cleanup temp file
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)

# -------------------------
# Interface Functions
# -------------------------
def display_database_statistics(df: pd.DataFrame, title: str):
    """Display database statistics in a nice format"""
    st.subheader(f"Database Statistics - {title}")
    if df.empty:
        st.info(f"No records found in {title.lower()}")
        return
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'severity' in df.columns:
            high_priority = len(df[df['severity'].isin(['High', 'Critical'])])
            st.metric("High/Critical Priority", high_priority)
        else:
            st.metric("High Priority", "N/A")
    
    with col3:
        if 'status' in df.columns:
            pending = len(df[df['status'].isin(['Reported', 'In Progress'])])
            st.metric("Pending Repairs", pending)
        else:
            st.metric("Pending", "N/A")
# -------------------------
# Initialization
# -------------------------
def initialize_app():
    """Initialize the application with SQLite database"""
    try:
        initialize_database()
        return True
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        return False