# core_functions.py - COMPLETE WITH EMAIL NOTIFICATION ON VERIFICATION
"""
Core functionality for FixMyStreet AI Road Inspection System
ARCHITECTURE:
- One UPLOAD_ID per image/video (the "report")
- Multiple DETECTION_IDs per upload (the defects found)
- Work orders reference UPLOAD_IDs (which contain multiple detections)
- Email notifications sent when admin verifies fixed defects
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
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.cell.cell import MergedCell

from geopy.geocoders import Nominatim
from ultralytics import YOLO

import streamlit as st
from PIL import Image

# Import email functionality from separate module
# Import email functionality from separate module
from email_service import (
    generate_otp, send_otp_email, store_otp, verify_otp,
    send_notification_email, send_verification_notification,
    send_fixed_notification,  # ⭐ NEW: Import for "Fixed" status notifications
    is_valid_email, is_gmail_address
)

# -------------------------
# Constants and Configuration
# -------------------------
DB_FILE = os.path.join(os.getcwd(), "road_inspection.db")

# Create database directory if needed
os.makedirs(os.path.dirname(DB_FILE) if os.path.dirname(DB_FILE) else ".", exist_ok=True)

STATUS_OPTIONS = ["Reported", "In Progress", "Work Order Generated", "Fixed", "Verified", "Cancelled"]
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
    """Initialize SQLite database with upload-based architecture"""
    try:
        # Ensure DB directory exists (works in container & cloud)
        db_dir = os.path.dirname(DB_FILE)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Users table
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

        # UPLOADS table - One record per image/video (this is the "report")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location TEXT NOT NULL,
                road_name TEXT,
                gps_lat REAL,
                gps_lon REAL,
                severity TEXT NOT NULL,
                size_category TEXT,
                notes TEXT,
                inspector TEXT,
                inspector_email TEXT,
                status TEXT DEFAULT 'Reported',
                work_order_id TEXT,
                work_order_generated_at TIMESTAMP,
                date_fixed TEXT,
                assigned_to TEXT,
                file_type TEXT,
                detection_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # DETECTIONS table - Multiple detections per upload
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id TEXT UNIQUE NOT NULL,
                upload_id TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                repair_method TEXT,
                priority_score INTEGER,
                frame_number INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
            )
        ''')

        # OTP table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS otp_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_used BOOLEAN DEFAULT FALSE
            )
        ''')

        # WORK ORDERS table - References upload_ids (not detection_ids) with contractor info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_order_id TEXT UNIQUE NOT NULL,
                upload_ids TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                status TEXT DEFAULT 'Pending',
                assigned_to TEXT,
                contractor_name TEXT,
                notes TEXT,
                total_detections INTEGER DEFAULT 0
            )
        ''')

        # Migrate existing database if needed
        migrate_database(cursor)

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
        return True

    except Exception as e:
        # Show a concise error and return False so caller can handle it
        st.error(f"Database initialization error: {str(e)}")
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass
        return False

def migrate_database(cursor):
    """Migrate existing database to new schema"""
    try:
        # Check if uploads table exists and has required columns
        cursor.execute("PRAGMA table_info(uploads)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'uploads' in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            # Add detection_count column if missing
            if 'detection_count' not in columns:
                cursor.execute("ALTER TABLE uploads ADD COLUMN detection_count INTEGER DEFAULT 0")
                # Update existing records with actual detection counts
                cursor.execute("""
                    UPDATE uploads 
                    SET detection_count = (
                        SELECT COUNT(*) 
                        FROM detections 
                        WHERE detections.upload_id = uploads.upload_id
                    )
                """)

            # Add inspector_email column if missing
            if 'inspector_email' not in columns:
                cursor.execute("ALTER TABLE uploads ADD COLUMN inspector_email TEXT")
                # Try to populate from users table
                cursor.execute("""
                    UPDATE uploads 
                    SET inspector_email = (
                        SELECT email 
                        FROM users 
                        WHERE users.username = uploads.inspector
                    )
                """)

        # Check if work_orders table has contractor_name column
        cursor.execute("PRAGMA table_info(work_orders)")
        wo_columns = [col[1] for col in cursor.fetchall()]

        if 'work_orders' in [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            # Add contractor_name column if missing
            if 'contractor_name' not in wo_columns:
                cursor.execute("ALTER TABLE work_orders ADD COLUMN contractor_name TEXT")

            # Add total_detections column if missing
            if 'total_detections' not in wo_columns:
                cursor.execute("ALTER TABLE work_orders ADD COLUMN total_detections INTEGER DEFAULT 0")

    except Exception as e:
        # If migration fails, just continue - new tables will be created correctly
        pass

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
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    show_success_message("Successfully logged out!")
    safe_rerun()

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
# Upload and Detection Functions
# -------------------------
def save_upload_with_detections(upload_data: dict, detections_list: list) -> bool:
    """
    Save ONE upload record with MULTIPLE detections
    This is the core of the upload-based architecture
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        detection_count = len(detections_list)

        # Get inspector email from users table
        cursor.execute("SELECT email FROM users WHERE username = ?", (upload_data['inspector'],))
        email_result = cursor.fetchone()
        inspector_email = email_result[0] if email_result else None

        # Insert ONE upload record with inspector email
        cursor.execute('''
            INSERT INTO uploads (
                upload_id, timestamp, location, road_name, gps_lat, gps_lon,
                severity, size_category, notes, inspector, inspector_email, 
                status, file_type, detection_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            upload_data['upload_id'],
            upload_data['timestamp'],
            upload_data['location'],
            upload_data['road_name'],
            upload_data['gps_lat'],
            upload_data['gps_lon'],
            upload_data['severity'],
            upload_data['size_category'],
            upload_data['notes'],
            upload_data['inspector'],
            inspector_email,
            'Reported',
            upload_data.get('file_type', 'image'),
            detection_count
        ))

        # Insert MULTIPLE detections for this ONE upload
        for detection in detections_list:
            cursor.execute('''
                INSERT INTO detections (
                    detection_id, upload_id, defect_type, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    repair_method, priority_score, frame_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection['detection_id'],
                upload_data['upload_id'],
                detection['defect_type'],
                detection['confidence'],
                detection.get('bbox_x1', 0),
                detection.get('bbox_y1', 0),
                detection.get('bbox_x2', 0),
                detection.get('bbox_y2', 0),
                detection['repair_method'],
                detection['priority_score'],
                detection.get('frame_number', 0)
            ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"Error saving upload data: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def get_uploads_data(status_filter=None, inspector_filter=None):
    """Get uploads data with detection counts and defect types"""
    try:
        conn = get_db_connection()

        query = """
        SELECT 
            u.*,
            GROUP_CONCAT(DISTINCT d.defect_type) as defect_types
        FROM uploads u
        LEFT JOIN detections d ON u.upload_id = d.upload_id
        """

        params = []
        conditions = []

        if status_filter:
            if isinstance(status_filter, list):
                placeholders = ','.join(['?' for _ in status_filter])
                conditions.append(f"u.status IN ({placeholders})")
                params.extend(status_filter)
            else:
                conditions.append("u.status = ?")
                params.append(status_filter)

        if inspector_filter:
            conditions.append("u.inspector = ?")
            params.append(inspector_filter)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY u.upload_id ORDER BY u.timestamp DESC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    except Exception as e:
        st.error(f"Error retrieving uploads data: {str(e)}")
        return pd.DataFrame()

def get_detections_for_upload(upload_id: str):
    """Get all detections for a specific upload"""
    try:
        conn = get_db_connection()
        query = "SELECT * FROM detections WHERE upload_id = ? ORDER BY detection_id"
        df = pd.read_sql_query(query, conn, params=(upload_id,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error retrieving detections: {str(e)}")
        return pd.DataFrame()

def update_upload_status(upload_id: str, new_status: str, assigned_to: str = None):
    """
    Update upload status with email notification on verification
    ⭐ NEW: Sends email to inspector when status changes to "Verified"
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get current status and upload details before updating
        cursor.execute("""
            SELECT status, inspector_email, location, road_name, 
                   detection_count, timestamp, inspector
            FROM uploads 
            WHERE upload_id = ?
        """, (upload_id,))

        result = cursor.fetchone()
        if not result:
            conn.close()
            st.error(f"Upload {upload_id} not found")
            return False

        old_status, inspector_email, location, road_name, defect_count, reported_date, inspector_name = result

        # Update the status
        if assigned_to:
            cursor.execute(
                "UPDATE uploads SET status = ?, assigned_to = ?, updated_at = ? WHERE upload_id = ?",
                (new_status, assigned_to, datetime.now().isoformat(), upload_id)
            )
        else:
            cursor.execute(
                "UPDATE uploads SET status = ?, updated_at = ? WHERE upload_id = ?",
                (new_status, datetime.now().isoformat(), upload_id)
            )

        if new_status in ['Fixed', 'Verified']:
            cursor.execute(
                "UPDATE uploads SET date_fixed = ? WHERE upload_id = ?",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), upload_id)
            )

        conn.commit()
        conn.close()

        # ⭐ NEW: Send email notification when status changes from "Fixed" to "Verified"
        if old_status == "Fixed" and new_status == "Verified":
            if inspector_email and is_valid_email(inspector_email):
                try:
                    # Get admin phone from secrets if available
                    admin_phone = None
                    try:
                        admin_phone = st.secrets.get("admin", {}).get("phone", "Contact Admin")
                    except:
                        admin_phone = "Contact Admin"

                    # Send verification notification email
                    email_sent = send_verification_notification(
                        recipient_email=inspector_email,
                        upload_id=upload_id,
                        location=location,
                        road_name=road_name,
                        defect_count=defect_count,
                        reported_date=reported_date,
                        admin_phone=admin_phone
                    )

                    if email_sent:
                        st.success(f"✅ Verification email sent to {inspector_name} ({inspector_email})")
                    else:
                        st.warning(f"⚠️ Status updated but email notification failed for {inspector_email}")

                except Exception as email_error:
                    st.warning(f"⚠️ Status updated but email notification failed: {str(email_error)}")
            else:
                st.info(f"ℹ️ Status updated. No valid email found for inspector {inspector_name}")

        return True

    except Exception as e:
        st.error(f"Error updating upload status: {str(e)}")
        return False

# -------------------------
# Work Order Functions - Upload-Based with Contractor
# -------------------------
def generate_work_order(upload_ids: list, created_by: str, notes: str = "", contractor_name: str = "") -> str:
    """
    Generate work order for MULTIPLE UPLOADS with contractor assignment
    Each upload contains multiple detections
    NOTE: Does NOT change upload status - that's handled separately
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        work_order_id = f"WO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        upload_ids_str = ",".join(upload_ids)

        # Check if detection_count column exists
        cursor.execute("PRAGMA table_info(uploads)")
        columns = [col[1] for col in cursor.fetchall()]
        has_detection_count = 'detection_count' in columns

        # Calculate total detections across all uploads
        if has_detection_count:
            cursor.execute(f'''
                SELECT SUM(detection_count) FROM uploads 
                WHERE upload_id IN ({','.join(['?' for _ in upload_ids])})
            ''', upload_ids)
            total_detections = cursor.fetchone()[0] or 0
        else:
            # Fallback: count from detections table
            cursor.execute(f'''
                SELECT COUNT(*) FROM detections 
                WHERE upload_id IN ({','.join(['?' for _ in upload_ids])})
            ''', upload_ids)
            total_detections = cursor.fetchone()[0] or 0

        # Insert work order with contractor name
        cursor.execute('''
            INSERT INTO work_orders (work_order_id, upload_ids, created_by, notes, contractor_name, total_detections)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (work_order_id, upload_ids_str, created_by, notes, contractor_name, total_detections))

        # Update uploads to link to work order (but DON'T change status)
        for upload_id in upload_ids:
            cursor.execute('''
                UPDATE uploads 
                SET work_order_id = ?,
                    work_order_generated_at = ?
                WHERE upload_id = ?
            ''', (work_order_id, datetime.now().isoformat(), upload_id))

        conn.commit()
        conn.close()
        return work_order_id

    except Exception as e:
        st.error(f"Error generating work order: {str(e)}")
        return None

def get_work_order_details(work_order_id: str):
    """Get comprehensive work order details with all uploads and detections"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get work order info
        cursor.execute("SELECT * FROM work_orders WHERE work_order_id = ?", (work_order_id,))
        wo_row = cursor.fetchone()

        if not wo_row:
            return None

        wo_columns = [desc[0] for desc in cursor.description]
        wo_data = dict(zip(wo_columns, wo_row))

        # Get upload IDs from work order
        upload_ids = wo_data['upload_ids'].split(',')

        # Get all uploads with their detections
        uploads_data = []
        all_detections = []

        for upload_id in upload_ids:
            # Get upload info
            upload_df = pd.read_sql_query(
                "SELECT * FROM uploads WHERE upload_id = ?", 
                conn, 
                params=(upload_id,)
            )

            if not upload_df.empty:
                upload_info = upload_df.iloc[0].to_dict()

                # Get ALL detections for this upload
                detections_df = pd.read_sql_query(
                    "SELECT * FROM detections WHERE upload_id = ? ORDER BY detection_id",
                    conn,
                    params=(upload_id,)
                )

                upload_info['detections'] = detections_df.to_dict('records')
                uploads_data.append(upload_info)
                all_detections.extend(detections_df.to_dict('records'))

        wo_data['uploads'] = uploads_data
        wo_data['all_detections'] = all_detections

        conn.close()
        return wo_data

    except Exception as e:
        st.error(f"Error retrieving work order details: {str(e)}")
        return None

def download_work_order_excel(work_order_id: str):
    """
    Generate Excel file for work order
    Shows upload-based structure: each upload with its multiple detections
    """
    try:
        wo_details = get_work_order_details(work_order_id)

        if not wo_details:
            st.error("Work order not found")
            return None, None

        # Create Excel file
        filename = f"{work_order_id}.xlsx"
        wb = openpyxl.Workbook()

        # Styling
        header_fill = PatternFill(start_color="1565C0", end_color="1565C0", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=12)
        title_font = Font(bold=True, size=14, color="1565C0")

        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # ===== SHEET 1: SUMMARY =====
        ws_summary = wb.active
        ws_summary.title = "Work Order Summary"

        # Title
        ws_summary['A1'] = "WORK ORDER SUMMARY"
        ws_summary['A1'].font = Font(bold=True, size=16, color="1565C0")
        ws_summary.merge_cells('A1:D1')
        ws_summary['A1'].alignment = Alignment(horizontal="center")

        # Work Order Details
        ws_summary['A3'] = "Work Order ID:"
        ws_summary['A3'].font = Font(bold=True)
        ws_summary['B3'] = wo_details['work_order_id']

        ws_summary['A4'] = "Created At:"
        ws_summary['A4'].font = Font(bold=True)
        ws_summary['B4'] = wo_details['created_at']

        ws_summary['A5'] = "Created By:"
        ws_summary['A5'].font = Font(bold=True)
        ws_summary['B5'] = wo_details['created_by']

        ws_summary['A6'] = "Contractor:"
        ws_summary['A6'].font = Font(bold=True)
        ws_summary['B6'] = wo_details.get('contractor_name', 'Not Assigned')

        ws_summary['A7'] = "Status:"
        ws_summary['A7'].font = Font(bold=True)
        ws_summary['B7'] = wo_details['status']

        ws_summary['A8'] = "Total Uploads (Reports):"
        ws_summary['A8'].font = Font(bold=True)
        ws_summary['B8'] = len(wo_details['uploads'])

        ws_summary['A9'] = "Total Defects Detected:"
        ws_summary['A9'].font = Font(bold=True)
        ws_summary['B9'] = wo_details['total_detections']

        ws_summary['A10'] = "Notes:"
        ws_summary['A10'].font = Font(bold=True)
        ws_summary['B10'] = wo_details.get('notes', 'No notes')

        # Upload Summary Table
        ws_summary['A12'] = "UPLOAD SUMMARY"
        ws_summary['A12'].font = title_font

        upload_headers = ["Upload ID", "Location", "Road Name", "Severity", "Defects Found", "Status"]
        for col, header in enumerate(upload_headers, 1):
            cell = ws_summary.cell(13, col, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        for row_idx, upload in enumerate(wo_details['uploads'], 14):
            ws_summary.cell(row_idx, 1, upload['upload_id']).border = thin_border
            ws_summary.cell(row_idx, 2, upload['location']).border = thin_border
            ws_summary.cell(row_idx, 3, upload['road_name']).border = thin_border
            ws_summary.cell(row_idx, 4, upload['severity']).border = thin_border
            ws_summary.cell(row_idx, 5, upload['detection_count']).border = thin_border
            ws_summary.cell(row_idx, 6, upload['status']).border = thin_border

        # ===== SHEET 2: DETAILED UPLOADS =====
        ws_uploads = wb.create_sheet("Upload Details")

        ws_uploads['A1'] = "DETAILED UPLOAD INFORMATION"
        ws_uploads['A1'].font = Font(bold=True, size=14, color="1565C0")
        ws_uploads.merge_cells('A1:J1')
        ws_uploads['A1'].alignment = Alignment(horizontal="center")

        upload_headers = ["Upload ID", "Timestamp", "Location", "Road Name", "Severity", 
                         "Detection Count", "Status", "Inspector", "GPS Lat", "GPS Lon"]

        for col, header in enumerate(upload_headers, 1):
            cell = ws_uploads.cell(3, col, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        for row_idx, upload in enumerate(wo_details['uploads'], 4):
            ws_uploads.cell(row_idx, 1, upload['upload_id']).border = thin_border
            ws_uploads.cell(row_idx, 2, upload['timestamp']).border = thin_border
            ws_uploads.cell(row_idx, 3, upload['location']).border = thin_border
            ws_uploads.cell(row_idx, 4, upload['road_name']).border = thin_border
            ws_uploads.cell(row_idx, 5, upload['severity']).border = thin_border
            ws_uploads.cell(row_idx, 6, upload['detection_count']).border = thin_border
            ws_uploads.cell(row_idx, 7, upload['status']).border = thin_border
            ws_uploads.cell(row_idx, 8, upload['inspector']).border = thin_border
            ws_uploads.cell(row_idx, 9, upload['gps_lat']).border = thin_border
            ws_uploads.cell(row_idx, 10, upload['gps_lon']).border = thin_border

        # ===== SHEET 3: ALL DETECTIONS (Grouped by Upload) =====
        ws_detections = wb.create_sheet("All Detections")

        ws_detections['A1'] = "ALL DETECTED DEFECTS (Grouped by Upload)"
        ws_detections['A1'].font = Font(bold=True, size=14, color="1565C0")
        ws_detections.merge_cells('A1:H1')
        ws_detections['A1'].alignment = Alignment(horizontal="center")

        detection_headers = ["Upload ID", "Detection ID", "Defect Type", "Confidence", 
                           "Repair Method", "Priority Score", "Frame #", "Bounding Box"]

        for col, header in enumerate(detection_headers, 1):
            cell = ws_detections.cell(3, col, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        current_row = 4

        # Group detections by upload
        for upload in wo_details['uploads']:
            upload_id = upload['upload_id']
            upload_detections = upload['detections']

            if upload_detections:
                # Add upload header row
                ws_detections.cell(current_row, 1, f"📍 {upload_id} - {upload['location']}").font = Font(bold=True, color="1565C0")
                ws_detections.merge_cells(f'A{current_row}:H{current_row}')
                current_row += 1

                # Add detections for this upload
                for detection in upload_detections:
                    bbox = f"({detection['bbox_x1']:.1f}, {detection['bbox_y1']:.1f}, {detection['bbox_x2']:.1f}, {detection['bbox_y2']:.1f})"

                    ws_detections.cell(current_row, 1, upload_id).border = thin_border
                    ws_detections.cell(current_row, 2, detection['detection_id']).border = thin_border
                    ws_detections.cell(current_row, 3, detection['defect_type']).border = thin_border
                    ws_detections.cell(current_row, 4, round(detection['confidence'], 3)).border = thin_border
                    ws_detections.cell(current_row, 5, detection['repair_method']).border = thin_border
                    ws_detections.cell(current_row, 6, detection['priority_score']).border = thin_border
                    ws_detections.cell(current_row, 7, detection.get('frame_number', 0)).border = thin_border
                    ws_detections.cell(current_row, 8, bbox).border = thin_border

                    current_row += 1

                # Add spacing between uploads
                current_row += 1

        # ===== SHEET 4: DEFECT SUMMARY =====
        ws_summary_defects = wb.create_sheet("Defect Summary")

        ws_summary_defects['A1'] = "DEFECT TYPE SUMMARY"
        ws_summary_defects['A1'].font = Font(bold=True, size=14, color="1565C0")
        ws_summary_defects.merge_cells('A1:D1')
        ws_summary_defects['A1'].alignment = Alignment(horizontal="center")

        # Calculate defect statistics
        defect_stats = {}
        for detection in wo_details['all_detections']:
            defect_type = detection['defect_type']
            if defect_type not in defect_stats:
                defect_stats[defect_type] = {
                    'count': 0,
                    'total_confidence': 0,
                    'uploads': set()
                }
            defect_stats[defect_type]['count'] += 1
            defect_stats[defect_type]['total_confidence'] += detection['confidence']
            defect_stats[defect_type]['uploads'].add(detection['upload_id'])

        summary_headers = ["Defect Type", "Total Count", "Avg Confidence", "Affected Uploads"]
        for col, header in enumerate(summary_headers, 1):
            cell = ws_summary_defects.cell(3, col, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        for row_idx, (defect_type, stats) in enumerate(sorted(defect_stats.items()), 4):
            avg_confidence = stats['total_confidence'] / stats['count']

            ws_summary_defects.cell(row_idx, 1, defect_type).border = thin_border
            ws_summary_defects.cell(row_idx, 2, stats['count']).border = thin_border
            ws_summary_defects.cell(row_idx, 3, f"{avg_confidence:.2%}").border = thin_border
            ws_summary_defects.cell(row_idx, 4, len(stats['uploads'])).border = thin_border

        # Auto-adjust column widths for all sheets
        for ws in [ws_summary, ws_uploads, ws_detections, ws_summary_defects]:
            for col_num in range(1, ws.max_column + 1):
                column_letter = openpyxl.utils.get_column_letter(col_num)
                max_length = 0

                for row_num in range(1, ws.max_row + 1):
                    cell = ws.cell(row_num, col_num)
                    # Skip merged cells
                    if isinstance(cell, openpyxl.cell.cell.MergedCell):
                        continue
                    try:
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass

                if max_length > 0:
                    adjusted_width = min(max_length + 2, 50)
                    ws.column_dimensions[column_letter].width = adjusted_width

        # Save to bytes
        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)

        return excel_buffer, filename

    except Exception as e:
        st.error(f"Error generating work order Excel: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def download_work_order_html(work_order_id: str):
    """
    Generate HTML file for work order with contractor information and location details
    Professional format suitable for printing and sharing
    """
    try:
        wo_details = get_work_order_details(work_order_id)

        if not wo_details:
            st.error("Work order not found")
            return None, None

        filename = f"{work_order_id}.html"

        # Build HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Work Order - {work_order_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        .header {{
            border-bottom: 4px solid #1565C0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #1565C0;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 18px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .info-item {{
            padding: 10px;
        }}
        
        .info-item .label {{
            font-weight: bold;
            color: #1565C0;
            display: block;
            margin-bottom: 5px;
        }}
        
        .info-item .value {{
            color: #333;
            font-size: 16px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            color: #1565C0;
            font-size: 24px;
            border-bottom: 2px solid #1565C0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        table thead {{
            background: #1565C0;
            color: white;
        }}
        
        table th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        table tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        .upload-card {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }}
        
        .upload-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #1565C0;
        }}
        
        .upload-card-title {{
            font-size: 20px;
            font-weight: bold;
            color: #1565C0;
        }}
        
        .upload-card-info {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        
        .badge-critical {{
            background: #f44336;
            color: white;
        }}
        
        .badge-high {{
            background: #ff9800;
            color: white;
        }}
        
        .badge-medium {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge-low {{
            background: #4caf50;
            color: white;
        }}
        
        .detection-list {{
            background: white;
            border-radius: 4px;
            padding: 15px;
        }}
        
        .detection-item {{
            padding: 10px;
            border-left: 4px solid #1565C0;
            margin-bottom: 10px;
            background: #f5f5f5;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #1565C0 0%, #0d47a1 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-card .number {{
            font-size: 36px;
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }}
        
        .stat-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #666;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
            
            .upload-card {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛣️ FixMyStreet Work Order</h1>
            <div class="subtitle">Road Defect Repair Work Order</div>
        </div>
        
        <div class="info-grid">
            <div class="info-item">
                <span class="label">Work Order ID:</span>
                <span class="value">{wo_details['work_order_id']}</span>
            </div>
            <div class="info-item">
                <span class="label">Created Date:</span>
                <span class="value">{wo_details['created_at']}</span>
            </div>
            <div class="info-item">
                <span class="label">Created By:</span>
                <span class="value">{wo_details['created_by']}</span>
            </div>
            <div class="info-item">
                <span class="label">Status:</span>
                <span class="value">{wo_details['status']}</span>
            </div>
            <div class="info-item">
                <span class="label">Contractor Assigned:</span>
                <span class="value" style="font-weight: bold; color: #1565C0;">{wo_details.get('contractor_name', 'Not Assigned')}</span>
            </div>
            <div class="info-item">
                <span class="label">Notes:</span>
                <span class="value">{wo_details.get('notes', 'No notes provided')}</span>
            </div>
        </div>
        
        <div class="summary-stats">
            <div class="stat-card">
                <span class="number">{len(wo_details['uploads'])}</span>
                <span class="label">Total Reports</span>
            </div>
            <div class="stat-card">
                <span class="number">{wo_details['total_detections']}</span>
                <span class="label">Total Defects</span>
            </div>
            <div class="stat-card">
                <span class="number">{len(set(u['location'] for u in wo_details['uploads']))}</span>
                <span class="label">Locations</span>
            </div>
            <div class="stat-card">
                <span class="number">{len(set(u['road_name'] for u in wo_details['uploads']))}</span>
                <span class="label">Roads</span>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📍 Upload Reports & Locations</h2>
"""
        
        # Add each upload as a card with location details
        for upload in wo_details['uploads']:
            severity_class = upload['severity'].lower()
            
            html_content += f"""
            <div class="upload-card">
                <div class="upload-card-header">
                    <div class="upload-card-title">{upload['upload_id']}</div>
                    <span class="badge badge-{severity_class}">{upload['severity']}</span>
                </div>
                
                <div class="upload-card-info">
                    <div>
                        <strong>📍 Location:</strong><br>
                        {upload['location']}
                    </div>
                    <div>
                        <strong>🛣️ Road Name:</strong><br>
                        {upload['road_name']}
                    </div>
                    <div>
                        <strong>📅 Reported:</strong><br>
                        {upload['timestamp']}
                    </div>
                    <div>
                        <strong>👤 Inspector:</strong><br>
                        {upload['inspector']}
                    </div>
                    <div>
                        <strong>🔍 Defects Found:</strong><br>
                        {upload['detection_count']} defects
                    </div>
                    <div>
                        <strong>📊 Status:</strong><br>
                        {upload['status']}
                    </div>
                </div>
                
                <div style="margin-top: 10px;">
                    <strong>🌍 GPS Coordinates:</strong> 
                    Lat: {upload['gps_lat']:.6f}, Lon: {upload['gps_lon']:.6f}
                    <a href="https://www.google.com/maps?q={upload['gps_lat']},{upload['gps_lon']}" 
                       target="_blank" style="color: #1565C0; margin-left: 10px;">
                       View on Google Maps →
                    </a>
                </div>
                
                <div class="detection-list">
                    <strong style="display: block; margin-bottom: 10px; color: #1565C0;">
                        Detected Defects:
                    </strong>
"""
            
            # Add detections for this upload
            for detection in upload['detections']:
                html_content += f"""
                    <div class="detection-item">
                        <strong>{detection['detection_id']}</strong> - 
                        {detection['defect_type']} 
                        (Confidence: {detection['confidence']:.1%})
                        <br>
                        <small>Repair Method: {detection['repair_method']} | 
                        Priority Score: {detection['priority_score']}</small>
                    </div>
"""
            
            html_content += """
                </div>
            </div>
"""
        
        # Add defect summary table
        defect_stats = {}
        for detection in wo_details['all_detections']:
            defect_type = detection['defect_type']
            if defect_type not in defect_stats:
                defect_stats[defect_type] = {
                    'count': 0,
                    'total_confidence': 0,
                    'uploads': set()
                }
            defect_stats[defect_type]['count'] += 1
            defect_stats[defect_type]['total_confidence'] += detection['confidence']
            defect_stats[defect_type]['uploads'].add(detection['upload_id'])
        
        html_content += """
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Defect Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Defect Type</th>
                        <th>Total Count</th>
                        <th>Average Confidence</th>
                        <th>Affected Uploads</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for defect_type, stats in sorted(defect_stats.items()):
            avg_confidence = stats['total_confidence'] / stats['count']
            html_content += f"""
                    <tr>
                        <td><strong>{defect_type}</strong></td>
                        <td>{stats['count']}</td>
                        <td>{avg_confidence:.1%}</td>
                        <td>{len(stats['uploads'])}</td>
                    </tr>
"""
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p><strong>FixMyStreet AI Road Inspection System</strong></p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 10px; color: #1565C0;">
                Contractor: <strong>{wo_details.get('contractor_name', 'Not Assigned')}</strong>
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        # Convert to bytes
        html_bytes = html_content.encode('utf-8')
        
        return html_bytes, filename
        
    except Exception as e:
        st.error(f"Error generating HTML work order: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def delete_upload(upload_id: str):
    """Delete upload and all its detections (CASCADE)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete detections first (or rely on CASCADE)
        cursor.execute("DELETE FROM detections WHERE upload_id = ?", (upload_id,))
        
        # Delete upload
        cursor.execute("DELETE FROM uploads WHERE upload_id = ?", (upload_id,))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error deleting upload: {str(e)}")
        return False

def get_repair_statistics():
    """Get repair statistics from uploads table with error handling"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if detection_count column exists
        cursor.execute("PRAGMA table_info(uploads)")
        columns = [col[1] for col in cursor.fetchall()]
        has_detection_count = 'detection_count' in columns
        
        # Get status counts
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM uploads 
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Get severity counts
        cursor.execute("""
            SELECT severity, COUNT(*) as count 
            FROM uploads 
            GROUP BY severity
        """)
        severity_counts = dict(cursor.fetchall())
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM uploads")
        total_count = cursor.fetchone()[0]
        
        # Get total detections
        if has_detection_count:
            cursor.execute("SELECT SUM(detection_count) FROM uploads")
            total_detections = cursor.fetchone()[0] or 0
        else:
            # Fallback: count from detections table
            cursor.execute("""
                SELECT COUNT(*) FROM detections 
                WHERE upload_id IN (SELECT upload_id FROM uploads)
            """)
            total_detections = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total': total_count,
            'total_detections': int(total_detections),
            'status_counts': status_counts,
            'severity_counts': severity_counts,
            'active': status_counts.get('Reported', 0) + status_counts.get('In Progress', 0),
            'completed': status_counts.get('Fixed', 0) + status_counts.get('Verified', 0)
        }
        
    except Exception as e:
        st.error(f"Error getting repair statistics: {str(e)}")
        return {'total': 0, 'total_detections': 0, 'status_counts': {}, 'severity_counts': {}, 'active': 0, 'completed': 0}


# -------------------------
# Duplicate Detection Functions
# -------------------------
def calculate_gps_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates in meters using Haversine formula
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of Earth in meters
    r = 6371000
    
    return c * r

def check_duplicate_location(lat, lon, radius_meters=5):
    """
    Check if there's an existing report within the specified radius
    Returns: (is_duplicate, duplicate_records_list)
    """
    try:
        conn = get_db_connection()
        
        # Get all uploads with GPS coordinates
        query = """
        SELECT u.upload_id, u.location, u.road_name, u.gps_lat, u.gps_lon, 
               u.severity, u.status, u.detection_count, u.inspector, u.timestamp,
               GROUP_CONCAT(DISTINCT d.defect_type) as defect_types
        FROM uploads u
        LEFT JOIN detections d ON u.upload_id = d.upload_id
        WHERE u.gps_lat IS NOT NULL AND u.gps_lon IS NOT NULL
        GROUP BY u.upload_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return False, []
        
        # Calculate distances for all records
        duplicates = []
        for idx, row in df.iterrows():
            distance = calculate_gps_distance(lat, lon, row['gps_lat'], row['gps_lon'])
            if distance <= radius_meters:
                duplicates.append({
                    'upload_id': row['upload_id'],
                    'location': row['location'],
                    'road_name': row['road_name'],
                    'gps_lat': row['gps_lat'],
                    'gps_lon': row['gps_lon'],
                    'distance': round(distance, 2),
                    'severity': row['severity'],
                    'status': row['status'],
                    'detection_count': row['detection_count'],
                    'inspector': row['inspector'],
                    'timestamp': row['timestamp'],
                    'defect_types': row['defect_types']
                })
        
        # Sort by distance
        duplicates = sorted(duplicates, key=lambda x: x['distance'])
        
        return len(duplicates) > 0, duplicates
        
    except Exception as e:
        st.error(f"Error checking for duplicates: {str(e)}")
        return False, []

def display_duplicate_warning(duplicates, current_lat, current_lon):
    """
    Display warning page when duplicate location is detected
    """
    st.warning("⚠️ **Duplicate Location Detected!**")
    
    st.markdown(f"""
    ### 📍 Similar Reports Found Nearby
    
    We found **{len(duplicates)}** existing report(s) within 5 meters of your location.
    
    **Your Location:** Latitude {current_lat:.6f}, Longitude {current_lon:.6f}
    """)
    
    # Display each duplicate
    for i, dup in enumerate(duplicates, 1):
        with st.expander(f"📋 Report #{i} - {dup['upload_id']} ({dup['distance']}m away)", expanded=(i == 1)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Report Details:**")
                st.write(f"🆔 Upload ID: `{dup['upload_id']}`")
                st.write(f"📍 Location: {dup['location']}")
                st.write(f"🛣️ Road: {dup['road_name']}")
                st.write(f"👤 Inspector: {dup['inspector']}")
                st.write(f"📅 Reported: {dup['timestamp']}")
            
            with col2:
                st.markdown("**Status Information:**")
                
                # Status badge with color
                status_colors = {
                    'Reported': '🔴',
                    'In Progress': '🟡',
                    'Work Order Generated': '🟠',
                    'Fixed': '🟢',
                    'Verified': '✅',
                    'Cancelled': '⚫'
                }
                status_icon = status_colors.get(dup['status'], '⚪')
                st.write(f"{status_icon} Status: **{dup['status']}**")
                
                # Severity badge
                severity_colors = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                severity_icon = severity_colors.get(dup['severity'], '⚪')
                st.write(f"{severity_icon} Severity: **{dup['severity']}**")
                
                st.write(f"🔍 Defects Found: **{dup['detection_count']}**")
                if dup['defect_types']:
                    st.write(f"🏷️ Types: {dup['defect_types']}")
            
            # Map link
            st.markdown(f"""
            <a href="https://www.google.com/maps?q={dup['gps_lat']},{dup['gps_lon']}" 
               target="_blank" style="color: #1565C0; text-decoration: none;">
               🗺️ View on Google Maps →
            </a>
            """, unsafe_allow_html=True)
            
            # Show detailed detections
            st.markdown("**Detected Defects:**")
            detections_df = get_detections_for_upload(dup['upload_id'])
            if not detections_df.empty:
                display_cols = ['detection_id', 'defect_type', 'confidence', 'repair_method']
                available_cols = [col for col in display_cols if col in detections_df.columns]
                st.dataframe(detections_df[available_cols], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Options for the user
    st.markdown("### 🤔 What would you like to do?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Option 1: Cancel Upload**")
        st.info("If this is the same defect that's already been reported, you can cancel your upload.")
        if st.button("❌ Cancel My Upload", key="cancel_upload", use_container_width=True):
            st.session_state.pop('duplicate_check_failed', None)
            st.session_state.pop('pending_upload_data', None)
            show_success_message("Upload cancelled. The existing reports will handle this location.")
            safe_rerun()
    
    with col2:
        st.markdown("**Option 2: Add Additional Report**")
        st.info("If this is a different or additional defect at the same location, you can proceed.")
        if st.button("➕ Proceed with Upload", key="proceed_upload", use_container_width=True):
            st.session_state['duplicate_override'] = True
            show_success_message("Proceeding with upload despite nearby reports...")
            safe_rerun()
    
    with col3:
        st.markdown("**Option 3: Update Existing**")
        st.info("Contact the original inspector or admin to update the existing report.")
        st.button("📞 Contact Support", key="contact_support", use_container_width=True, disabled=True)

# -------------------------
# Detection Interface
# -------------------------
def detection_interface(model, geolocator):
    """Detection interface for upload-based system with duplicate detection"""
    st.header("AI Road Defect Detection")
    
    if not model:
        st.error("YOLO model not available. Please ensure 'best.pt' is in the application directory.")
        return
    
    # Check if we're in duplicate warning state
    if st.session_state.get('duplicate_check_failed', False) and not st.session_state.get('duplicate_override', False):
        pending_data = st.session_state.get('pending_upload_data', {})
        if pending_data:
            display_duplicate_warning(
                pending_data.get('duplicates', []),
                pending_data.get('lat'),
                pending_data.get('lon')
            )
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
            
            # Check for duplicate locations (unless override is set)
            if not st.session_state.get('duplicate_override', False):
                is_duplicate, duplicates = check_duplicate_location(lat, lon, radius_meters=5)
                
                if is_duplicate:
                    st.session_state['duplicate_check_failed'] = True
                    st.session_state['pending_upload_data'] = {
                        'duplicates': duplicates,
                        'lat': lat,
                        'lon': lon,
                        'location': location,
                        'road_name': road_name,
                        'severity': severity,
                        'size_category': size_category,
                        'notes': notes,
                        'uploaded_file': uploaded_file
                    }
                    st.warning(f"⚠️ Found {len(duplicates)} existing report(s) within 5 meters!")
                    safe_rerun()
                    return
            
            # If we reach here, either no duplicate or user chose to override
            if st.session_state.get('duplicate_override', False):
                st.info("ℹ️ Proceeding with upload despite nearby reports...")
                st.session_state.pop('duplicate_override', None)
                st.session_state.pop('duplicate_check_failed', None)
                st.session_state.pop('pending_upload_data', None)
            
            try:
                # Generate ONE unique upload ID per image/video
                upload_id = f"UPLOAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
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
                        
                        # Process and save: ONE upload, MULTIPLE detections
                        upload_data, detections_list, detection_count = process_image_detections_new(
                            prediction, upload_id, location, road_name, lat, lon, 
                            severity, size_category, notes, model
                        )
                        
                        if upload_data and detections_list:
                            if save_upload_with_detections(upload_data, detections_list):
                                st.success(f"✅ Successfully saved!")
                                st.info(f"**Upload ID (Report):** {upload_id}")
                                st.info(f"**Defects Detected:** {detection_count}")
                                
                                # Show detection summary
                                st.subheader("Detection Summary")
                                summary_df = pd.DataFrame(detections_list)
                                st.dataframe(summary_df[['detection_id', 'defect_type', 'confidence', 'repair_method']], 
                                           use_container_width=True)
                            else:
                                st.error("Failed to save detections to database")
                        else:
                            st.warning("No defects detected in the image")
                    else:
                        st.warning("No defects detected in the image")
                
                elif file_type.startswith('video'):
                    # Process video
                    st.info("Processing video... This may take a while.")
                    
                    try:
                        output_path, upload_data, detections_list, detection_count = process_video_detections_new(
                            temp_path, upload_id, location, road_name, lat, lon, 
                            severity, size_category, notes, model
                        )
                        
                        if output_path and upload_data and detections_list:
                            if save_upload_with_detections(upload_data, detections_list):
                                st.success(f"✅ Video processed successfully!")
                                st.info(f"**Upload ID (Report):** {upload_id}")
                                st.info(f"**Total Defects Detected:** {detection_count}")
                                
                                # Show detection summary
                                st.subheader("Detection Summary")
                                summary_df = pd.DataFrame(detections_list)
                                
                                # Group by defect type
                                defect_summary = summary_df.groupby('defect_type').agg({
                                    'detection_id': 'count',
                                    'confidence': 'mean'
                                }).rename(columns={'detection_id': 'count', 'confidence': 'avg_confidence'})
                                
                                st.dataframe(defect_summary, use_container_width=True)
                                
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
                                st.error("Failed to save detections to database")
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
# Display Functions
# -------------------------
def display_database_statistics(df: pd.DataFrame, title: str):
    """Display database statistics for uploads"""
    st.subheader(f"Statistics - {title}")
    if df.empty:
        st.info(f"No records found in {title.lower()}")
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", len(df))
    
    with col2:
        if 'detection_count' in df.columns:
            total_detections = df['detection_count'].sum()
            st.metric("Total Defects", int(total_detections))
        else:
            st.metric("Total Defects", "N/A")
    
    with col3:
        if 'severity' in df.columns:
            high_priority = len(df[df['severity'].isin(['High', 'Critical'])])
            st.metric("High/Critical", high_priority)
        else:
            st.metric("High Priority", "N/A")
    
    with col4:
        if 'status' in df.columns:
            pending = len(df[df['status'].isin(['Reported', 'In Progress'])])
            st.metric("Pending", pending)
        else:
            st.metric("Pending", "N/A")

# -------------------------
# User Authentication
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
        
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            st.error("Username (Email) already exists.")
            conn.close()
            return False
        
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
# Login Interface
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
                new_email = st.text_input("Gmail Address", key="reg_email")
                new_name = st.text_input("Full Name", key="reg_name")
                new_password = st.text_input("Choose a Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")

                st.markdown("---")
                st.markdown("##### Step 2: Verify Your Email")
                
                col_otp_btn, col_otp_field = st.columns([1, 2])

                with col_otp_btn:
                    if st.form_submit_button("✉️ Send OTP"):
                        if not is_gmail_address(new_email):
                            st.error("Please enter a valid Gmail address.")
                        else:
                            otp = generate_otp()
                            if send_otp_email(new_email, otp):
                                if store_otp(new_email, otp):
                                    st.session_state['otp_sent'] = True
                                    st.session_state['otp_email'] = new_email
                                    st.session_state['otp_time'] = datetime.now()
                                    st.info("OTP sent to your email.")

                with col_otp_field:
                    otp_input = st.text_input("Enter OTP", key="reg_otp", max_chars=6)

                st.markdown("---")
                register_submitted = st.form_submit_button("✅ Register Account")

                if register_submitted:
                    otp_sent = st.session_state.get('otp_sent', False)
                    otp_email_in_session = st.session_state.get('otp_email')
                    otp_time_in_session = st.session_state.get('otp_time')

                    if new_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif not all([new_email, new_name, new_password, otp_input]):
                        st.error("All fields required!")
                    elif not otp_sent or otp_email_in_session != new_email:
                        st.error("Please send OTP first.")
                    elif otp_time_in_session and (datetime.now() - otp_time_in_session) > timedelta(minutes=5):
                        st.error("OTP expired.")
                    elif not verify_otp(new_email, otp_input):
                        st.error("Invalid OTP.")
                    else:
                        if register_user(new_email, new_password, new_name):
                            st.success("Registration successful! Please log in.")
                            st.session_state.pop('otp_sent', None)
                            st.session_state.pop('otp_email', None)
                            st.session_state.pop('otp_time', None)

# -------------------------
# YOLO Detection Processing
# -------------------------
def extract_box_information(box, model_obj):
    """Safely extract class name and confidence from YOLO detection box"""
    try:
        cls_val = None
        conf_val = None
        
        if hasattr(box, 'cls'):
            cls_val = box.cls
        elif hasattr(box, 'class_id'):
            cls_val = box.class_id
            
        if hasattr(box, 'conf'):
            conf_val = box.conf
        elif hasattr(box, 'confidence'):
            conf_val = box.confidence
        
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

def extract_bbox_coords(box):
    """Extract bounding box coordinates"""
    try:
        if hasattr(box, 'xyxy'):
            coords = box.xyxy[0].cpu().numpy()
            return float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
        elif hasattr(box, 'boxes'):
            coords = box.boxes[0]
            return float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
        else:
            return 0.0, 0.0, 0.0, 0.0
    except:
        return 0.0, 0.0, 0.0, 0.0

def process_image_detections_new(prediction, upload_id: str, location: str, road_name: str, 
                                 lat: float, lon: float, severity: str, size_category: str, 
                                 notes: str, model_obj) -> tuple:
    """
    Process YOLO detections from image
    Returns: (upload_data, detections_list, detection_count)
    ONE upload_id with MULTIPLE detection_ids
    """
    if not model_obj or not prediction:
        return None, [], 0
        
    detections_list = []
    
    try:
        if len(prediction) > 0 and prediction[0].boxes is not None:
            boxes = prediction[0].boxes
            
            for i, box in enumerate(boxes):
                class_name, confidence = extract_box_information(box, model_obj)
                x1, y1, x2, y2 = extract_bbox_coords(box)
                repair_method = get_repair_method(class_name, severity)
                priority_score = calculate_priority_score(severity, class_name, size_category)
                
                detection_data = {
                    'detection_id': f"{upload_id}_DET{i+1:03d}",
                    'defect_type': class_name,
                    'confidence': round(confidence, 3),
                    'bbox_x1': round(x1, 2),
                    'bbox_y1': round(y1, 2),
                    'bbox_x2': round(x2, 2),
                    'bbox_y2': round(y2, 2),
                    'repair_method': repair_method,
                    'priority_score': priority_score,
                    'frame_number': 0
                }
                
                detections_list.append(detection_data)
        
        # Create upload data
        upload_data = {
            'upload_id': upload_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': location,
            'road_name': road_name,
            'gps_lat': round(float(lat), 6),
            'gps_lon': round(float(lon), 6),
            'severity': severity,
            'size_category': size_category,
            'notes': notes,
            'inspector': st.session_state.user_name,
            'file_type': 'image'
        }
        
        return upload_data, detections_list, len(detections_list)
                    
    except Exception as e:
        st.error(f"Error processing image detections: {str(e)}")
        return None, [], 0

def process_video_detections_new(video_path: str, upload_id: str, location: str, road_name: str, 
                                lat: float, lon: float, severity: str, size_category: str, 
                                notes: str, model_obj) -> tuple:
    """
    Process YOLO detections from video
    Returns: (output_path, upload_data, detections_list, detection_count)
    ONE upload_id with MULTIPLE detection_ids (sampled from frames)
    """
    if not model_obj:
        return None, None, [], 0
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None, [], 0
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    output_path = f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    detections_list = []
    frame_count = 0
    detection_counter = 0
    
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
                if frame_count % 30 == 0 and prediction and len(prediction) > 0:
                    if prediction[0].boxes is not None:
                        boxes = prediction[0].boxes
                        
                        for i, box in enumerate(boxes):
                            class_name, confidence = extract_box_information(box, model_obj)
                            x1, y1, x2, y2 = extract_bbox_coords(box)
                            repair_method = get_repair_method(class_name, severity)
                            priority_score = calculate_priority_score(severity, class_name, size_category)
                            
                            detection_counter += 1
                            detection_data = {
                                'detection_id': f"{upload_id}_DET{detection_counter:03d}",
                                'defect_type': class_name,
                                'confidence': round(confidence, 3),
                                'bbox_x1': round(x1, 2),
                                'bbox_y1': round(y1, 2),
                                'bbox_x2': round(x2, 2),
                                'bbox_y2': round(y2, 2),
                                'repair_method': repair_method,
                                'priority_score': priority_score,
                                'frame_number': frame_count
                            }
                            detections_list.append(detection_data)
                
                # Create annotated frame
                if prediction and len(prediction) > 0:
                    annotated_frame = prediction[0].plot()
                    if annotated_frame is not None:
                        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    else:
                        out.write(frame)
                else:
                    out.write(frame)
                    
            except Exception as e:
                st.warning(f"Error processing frame {frame_count}: {str(e)}")
                out.write(frame)
                continue
        
        # Cleanup
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        # Create upload data
        upload_data = {
            'upload_id': upload_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': location,
            'road_name': road_name,
            'gps_lat': round(float(lat), 6),
            'gps_lon': round(float(lon), 6),
            'severity': severity,
            'size_category': size_category,
            'notes': notes,
            'inspector': st.session_state.user_name,
            'file_type': 'video'
        }
        
        return output_path, upload_data, detections_list, len(detections_list)
        
    except Exception as e:
        cap.release()
        if 'out' in locals():
            out.release()
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Error processing video: {e}")