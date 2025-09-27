
"""
Admin Interface Module for FixMyStreet AI Road Inspection System
Handles admin functionality - database management, reports, and full system control
UI updated with proper button placement and admin panel merged into Settings
"""

import streamlit as st
import streamlit_option_menu as option_menu
import pandas as pd
from datetime import datetime
import sqlite3
import base64
import os
from PIL import Image
import io

from core_functions import (
    authenticate_user, register_user, initialize_database,
    show_success_message, safe_rerun, display_session_messages,
    load_yolo_model, initialize_geocoder,
    detection_interface, get_repairs_data, get_repair_statistics,
    update_repair_status, delete_repair, display_database_statistics,
    initialize_app, logout, DB_FILE, login_interface
)

# -------------------------
# Utility Functions
# -------------------------
def _embed_local_image_base64(path: str, crop_ratio: float = 0.7):
    """Embed local image file as base64."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith('.svg'):
        with open(path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode('utf-8')
        img_tag = f"<img src='data:image/svg+xml;base64,{b64}' style='height:60px; margin-right:20px;'/>"
        return img_tag, 'image/svg+xml'
    else:
        img = Image.open(path)
        cropped = img.crop((0, 0, img.width, int(img.height * crop_ratio)))
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_tag = f"<img src='data:image/png;base64,{b64}' style='height:60px; margin-right:20px;'/>"
        return img_tag, 'image/png'

# -------------------------
# Admin Dashboard
# -------------------------
def admin_dashboard():
    st.header("Administrative Dashboard")
    all_repairs = get_repairs_data()

    if not all_repairs.empty:
        stats = get_repair_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reports", stats['total'])
        with col2:
            st.metric("Active Reports", stats['active'])
        with col3:
            st.metric("Completed", stats['completed'])
        with col4:
            pending_pct = (stats['active'] / stats['total'] * 100) if stats['total'] > 0 else 0
            st.metric("Pending %", f"{pending_pct:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            if 'severity' in all_repairs.columns:
                st.subheader("Severity Distribution")
                severity_counts = all_repairs['severity'].value_counts()
                st.bar_chart(severity_counts)
        with col2:
            if 'status' in all_repairs.columns:
                st.subheader("Status Distribution")
                status_counts = all_repairs['status'].value_counts()
                st.bar_chart(status_counts)

        st.subheader("Recent Activity")
        recent_repairs = all_repairs.sort_values('timestamp', ascending=False).head(10)
        display_columns = ['detection_id', 'timestamp', 'location', 'defect_type', 'severity', 'status', 'inspector']
        available_columns = [col for col in display_columns if col in recent_repairs.columns]
        st.dataframe(recent_repairs[available_columns], use_container_width=True)

        show_inprogress_potholes_cracks(all_repairs)
    else:
        st.info("No inspection reports in the database yet.")

def show_inprogress_potholes_cracks(df):
    if df.empty or 'status' not in df.columns or 'defect_type' not in df.columns:
        return
    filtered = df[(df['status'] == 'In Progress') & (df['defect_type'].isin(['Pothole', 'Crack']))]
    if not filtered.empty:
        st.subheader("In Progress: Potholes & Cracks")
        st.dataframe(filtered, use_container_width=True)
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="Download In Progress Potholes & Cracks CSV",
            data=csv,
            file_name=f"inprogress_potholes_cracks_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

# -------------------------
# Database Table
# -------------------------
def display_database_table_with_workflow(df: pd.DataFrame, title: str, current_status: str):
    if df.empty:
        st.info(f"No data available in {title}")
        return
    st.subheader(f"{title}")
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_options = df['status'].unique() if 'status' in df.columns else []
        status_filter = st.multiselect("Filter by Status", options=status_options, default=status_options)
    with col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=df['severity'].unique() if 'severity' in df.columns else [],
            default=df['severity'].unique() if 'severity' in df.columns else []
        )
    with col3:
        defect_filter = st.multiselect(
            "Filter by Defect Type",
            options=df['defect_type'].unique() if 'defect_type' in df.columns else [],
            default=df['defect_type'].unique() if 'defect_type' in df.columns else []
        )
    filtered_df = df.copy()
    if status_filter and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    if severity_filter and 'severity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
    if defect_filter and 'defect_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['defect_type'].isin(defect_filter)]
    if 'detection_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['detection_id'].notna()]
    st.dataframe(filtered_df, use_container_width=True)
    export_status_csvs(filtered_df)
    if not filtered_df.empty:
        st.subheader("Workflow Management")
        selected_records = st.multiselect(
            "Select records for workflow actions:",
            options=filtered_df['detection_id'].tolist() if 'detection_id' in filtered_df.columns else []
        )
        if selected_records:
            create_workflow_buttons(current_status, selected_records, title)

def create_workflow_buttons(current_status: str, selected_records: list, title: str):
    if current_status == "Reported":
        if st.button("Move to In Progress"):
            update_records_status(selected_records, "In Progress")
            safe_rerun()
        if st.button("Generate Work Order"):
            update_records_status(selected_records, "Work Order Generated")
            safe_rerun()
        if st.button("Cancel"):
            update_records_status(selected_records, "Cancelled")
            safe_rerun()
    elif current_status == "In Progress":
        if st.button("Back to Reported"):
            update_records_status(selected_records, "Reported")
            safe_rerun()
        if st.button("Mark as Fixed"):
            update_records_status(selected_records, "Fixed")
            safe_rerun()
        if st.button("Generate Work Order"):
            update_records_status(selected_records, "Work Order Generated")
            safe_rerun()
        if st.button("Cancel"):
            update_records_status(selected_records, "Cancelled")
            safe_rerun()
    elif current_status == "Fixed":
        if st.button("Back to In Progress"):
            update_records_status(selected_records, "In Progress")
            safe_rerun()
        if st.button("Mark as Verified"):
            update_records_status(selected_records, "Verified")
            safe_rerun()
        # No generate work order here

def update_records_status(record_ids: list, new_status: str) -> int:
    success_count = 0
    for record_id in record_ids:
        if update_repair_status(record_id, new_status):
            success_count += 1
    return success_count

def export_status_csvs(df):
    if 'status' not in df.columns:
        return
    status_types = df['status'].unique()
    for status in status_types:
        status_df = df[df['status'] == status]
        if not status_df.empty:
            csv = status_df.to_csv(index=False)
            st.download_button(
                label=f"Download {status} Repairs CSV",
                data=csv,
                file_name=f"{status.lower().replace(' ', '_')}_repairs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                key=f"download_{status.replace(' ', '_')}_csv"
            )

# -------------------------
# Admin Controls
# -------------------------
def admin_controls():
    st.subheader("Admin Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Database"):
            if clear_database():
                show_success_message("Database cleared successfully!")
                safe_rerun()
    with col2:
        if st.button("Reset Database"):
            if reset_database():
                show_success_message("Database reset successfully!")
                safe_rerun()
    with col3:
        if st.button("Generate Report"):
            generate_comprehensive_report()

def clear_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM repairs")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

def reset_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS repairs")
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS otp_tokens")
        conn.commit()
        conn.close()
        initialize_app()
        return True
    except Exception as e:
        st.error(f"Error resetting database: {e}")
        return False

def generate_comprehensive_report():
    try:
        all_repairs = get_repairs_data()
        if all_repairs.empty:
            st.warning("No data available for report generation.")
            return
        stats = get_repair_statistics()
        report_filename = f"FixMyStreet_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
            all_repairs.to_excel(writer, sheet_name='All_Repairs', index=False)
            summary_data = {
                'Metric': ['Total Reports', 'Active Reports', 'Completed Reports'],
                'Value': [stats['total'], stats['active'], stats['completed']]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        with open(report_filename, 'rb') as f:
            st.download_button(
                label="Download Report",
                data=f.read(),
                file_name=report_filename,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        os.remove(report_filename)
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

# -------------------------
# Database Maintenance
# -------------------------
def backup_database():
    try:
        import shutil
        backup_filename = f"backup_{DB_FILE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(DB_FILE, backup_filename)
        with open(backup_filename, 'rb') as f:
            st.download_button(
                label="Download Database Backup",
                data=f.read(),
                file_name=backup_filename,
                mime='application/octet-stream'
            )
        os.remove(backup_filename)
    except Exception as e:
        st.error(f"Error creating backup: {e}")

# -------------------------
# Sidebar Navigation
# -------------------------
def create_admin_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: #1565c0; border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='color: white; margin: 0;'>Administrator</h3>
            <p style='color: #e3f2fd; margin: 0; font-weight: bold;'>{st.session_state.user_name}</p>
            <small style='color: #bbdefb;'>System Control Panel</small>
        </div>
        """, unsafe_allow_html=True)
        selected = option_menu.option_menu(
            menu_title="Navigation",
            options=["Dashboard", "AI Detection", "Reported", "In Progress", "Fixed", "Settings"],
            icons=['house', 'camera', 'exclamation-triangle', 'clock-history', 'check-circle', 'sliders'],
            menu_icon="shield-check",
            default_index=0,
            styles={"nav-link-selected": {"background-color": "#1565c0"}}
        )
        st.markdown("---")
        model, _ = load_yolo_model()
        db_exists = os.path.exists(DB_FILE)
        st.markdown("### System Status")
        st.write("✅ YOLO Model: Ready" if model else "❌ YOLO Model: Missing")
        st.write("✅ Database: Connected" if db_exists else "❌ Database: Missing")
        st.markdown("---")
        try:
            stats = get_repair_statistics()
            st.markdown("### Quick Stats")
            st.write(f"📊 Total Reports: {stats['total']}")
            st.write(f"🔄 Active Reports: {stats['active']}")
            st.write(f"✅ Completed: {stats['completed']}")
        except:
            pass
        st.markdown("---")
        if st.button("🚪 Logout", key="admin_logout_btn"):
            logout()
        return selected

# -------------------------
# Main Admin Application
# -------------------------
def admin_main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.username = None
        st.session_state.user_id = None
    if not initialize_app():
        st.error("Failed to initialize application.")
        return
    display_session_messages()
    if not st.session_state.get('logged_in', False):
        login_interface()
        return
    if st.session_state.user_role != 'admin':
        st.error("This interface is for administrators only.")
        if st.button("Logout"):
            logout()
        return
    model, model_status = load_yolo_model()
    geolocator = initialize_geocoder()
    selected = create_admin_sidebar()
    if selected == "Dashboard":
        admin_dashboard()
    elif selected == "AI Detection":
        if model is None:
            st.error("AI model not loaded.")
            return
        detection_interface(model, geolocator)
    elif selected == "Reported":
        st.header("📋 Reported Issues")
        reported_repairs = get_repairs_data(status_filter="Reported")
        if not reported_repairs.empty:
            display_database_statistics(reported_repairs, "Reported Issues")
            display_database_table_with_workflow(reported_repairs, "Reported Issues", "Reported")
        else:
            st.info("No reported issues.")
    elif selected == "In Progress":
        st.header("⏳ In Progress Work")
        inprogress_repairs = get_repairs_data(status_filter="In Progress")
        if not inprogress_repairs.empty:
            display_database_statistics(inprogress_repairs, "In Progress Work")
            display_database_table_with_workflow(inprogress_repairs, "In Progress Work", "In Progress")
        else:
            st.info("No work in progress.")
    elif selected == "Fixed":
        st.header("✅ Fixed Issues")
        fixed_repairs = get_repairs_data(status_filter="Fixed")
        if not fixed_repairs.empty:
            display_database_statistics(fixed_repairs, "Fixed Issues")
            display_database_table_with_workflow(fixed_repairs, "Fixed Issues", "Fixed")
        else:
            st.info("No fixed issues.")
    elif selected == "Settings":
        st.header("⚙️ System Settings")
        st.subheader("Administrator Information")
        st.write(f"**Name:** {st.session_state.user_name}")
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Role:** {st.session_state.user_role}")
        st.subheader("Database Information")
        st.write(f"**Database:** SQLite ({DB_FILE})")
        st.write(f"**Model Status:** {model_status}")
        st.subheader("System Configuration")
        try:
            sender_email = st.secrets["email"]["sender_email"]
            st.success("Email configuration: Active")
            st.write(f"**Sender Email:** {sender_email}")
        except:
            st.error("Email configuration: Not configured")
        # Merge Admin Panel here
        st.markdown("---")
        with st.expander("🔧 Admin Controls", expanded=False):
            admin_controls()
        with st.expander("🔨 Database Maintenance", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Initialize Database"):
                    if initialize_database():
                        show_success_message("Database initialized!")
                        safe_rerun()
                if st.button("Backup Database"):
                    backup_database()
            with col2:
                pass

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="FixMyStreet - Admin Panel",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    admin_main()

