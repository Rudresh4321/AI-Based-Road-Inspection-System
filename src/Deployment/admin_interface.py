"""
Admin Interface Module for FixMyStreet AI Road Inspection System
UPDATED: Contractor assignment and HTML work order generation
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
    initialize_app, logout, DB_FILE, login_interface,
    generate_work_order, download_work_order_excel, get_detections_for_upload,
    get_uploads_data, download_work_order_html
)

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
            st.metric("Total Uploads", stats['total'])
        with col2:
            st.metric("Active Reports", stats['active'])
        with col3:
            st.metric("Completed", stats['completed'])
        with col4:
            st.metric("Total Detections", stats['total_detections'])

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
        display_columns = ['upload_id', 'timestamp', 'location', 'road_name', 'severity', 'status', 'detection_count', 'inspector']
        available_columns = [col for col in display_columns if col in recent_repairs.columns]
        st.dataframe(recent_repairs[available_columns], use_container_width=True)

    else:
        st.info("No inspection reports in the database yet.")

# -------------------------
# Database Table with Workflow - Enhanced with Contractor Assignment
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
        status_filter = st.multiselect("Filter by Status", options=status_options, default=status_options, key=f"status_filter_{title}")
    with col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=df['severity'].unique() if 'severity' in df.columns else [],
            default=df['severity'].unique() if 'severity' in df.columns else [],
            key=f"severity_filter_{title}"
        )
    with col3:
        if 'defect_types' in df.columns:
            all_defect_types = set()
            for types_str in df['defect_types'].dropna():
                all_defect_types.update(types_str.split(','))
            defect_filter = st.multiselect(
                "Filter by Defect Type",
                options=sorted(all_defect_types),
                default=sorted(all_defect_types),
                key=f"defect_filter_{title}"
            )
        else:
            defect_filter = []
    
    # Apply filters
    filtered_df = df.copy()
    if status_filter and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    if severity_filter and 'severity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
    if defect_filter and 'defect_types' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['defect_types'].apply(
            lambda x: any(dt in str(x) for dt in defect_filter) if pd.notna(x) else False
        )]
    
    if 'upload_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['upload_id'].notna()]
    
    # Display dataframe
    display_cols = ['upload_id', 'timestamp', 'location', 'road_name', 'severity', 'detection_count', 'status', 'inspector']
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    st.dataframe(filtered_df[available_cols], use_container_width=True)
    
    # Export CSVs
    export_status_csvs(filtered_df)
    
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("Workflow Management")
        
        # Select upload_ids
        selected_uploads = st.multiselect(
            "Select uploads for workflow actions:",
            options=filtered_df['upload_id'].tolist() if 'upload_id' in filtered_df.columns else [],
            key=f"selected_uploads_{title}"
        )
        
        if selected_uploads:
            # Show summary of selected uploads
            selected_df = filtered_df[filtered_df['upload_id'].isin(selected_uploads)]
            total_detections = selected_df['detection_count'].sum() if 'detection_count' in selected_df.columns else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Uploads", len(selected_uploads))
            with col2:
                st.metric("Total Detections", int(total_detections))
            with col3:
                st.metric("Avg Severity", selected_df['severity'].mode()[0] if not selected_df.empty else "N/A")
            
            # Show details of each selected upload
            with st.expander("View Selected Upload Details"):
                for upload_id in selected_uploads:
                    upload_info = filtered_df[filtered_df['upload_id'] == upload_id].iloc[0]
                    st.markdown(f"**{upload_id}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"📍 {upload_info['location']}")
                    with col2:
                        st.write(f"⚠️ {upload_info['severity']}")
                    with col3:
                        st.write(f"🔍 {upload_info['detection_count']} detections")
                    
                    # Show detections for this upload
                    detections_df = get_detections_for_upload(upload_id)
                    if not detections_df.empty:
                        st.dataframe(detections_df[['detection_id', 'defect_type', 'confidence', 'repair_method']], 
                                   use_container_width=True)
                    st.markdown("---")
            
            # Workflow buttons
            create_workflow_buttons(current_status, selected_uploads, title)

def create_workflow_buttons(current_status: str, selected_uploads: list, title: str):
    """
    Enhanced workflow with contractor assignment
    Work order generation only appears on In Progress tab with contractor name
    """
    
    if current_status == "Reported":
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Move to In Progress", key=f"btn_inprogress_{title}", use_container_width=True):
                update_records_status(selected_uploads, "In Progress")
                show_success_message(f"Moved {len(selected_uploads)} uploads to In Progress")
                safe_rerun()
        
        with col2:
            if st.button("❌ Cancel", key=f"btn_cancel_{title}", use_container_width=True):
                update_records_status(selected_uploads, "Cancelled")
                show_success_message(f"Cancelled {len(selected_uploads)} uploads")
                safe_rerun()
    
    elif current_status == "In Progress":
        # Contractor name input
        contractor_name = st.text_input(
            "Contractor Name (Required for Work Order)", 
            key=f"contractor_name_{title}",
            placeholder="Enter contractor name..."
        )
        
        # Work order notes input
        work_order_notes = st.text_input(
            "Work Order Notes (Optional)", 
            key=f"wo_notes_inprog_{title}",
            placeholder="Enter notes for work order..."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Back to Reported", key=f"btn_back_{title}", use_container_width=True):
                update_records_status(selected_uploads, "Reported")
                show_success_message(f"Moved {len(selected_uploads)} uploads back to Reported")
                safe_rerun()
        
        with col2:
            if st.button("✅ Mark as Fixed", key=f"btn_fixed_{title}", use_container_width=True):
                update_records_status(selected_uploads, "Fixed")
                show_success_message(f"Marked {len(selected_uploads)} uploads as Fixed")
                safe_rerun()
        
        with col3:
            # Generate Work Order with contractor name
            if st.button("📋 Generate Work Order", key=f"btn_wo_inprog_{title}", use_container_width=True):
                if not contractor_name or contractor_name.strip() == "":
                    st.error("⚠️ Please enter a contractor name before generating work order!")
                else:
                    wo_id = generate_work_order(selected_uploads, st.session_state.user_name, work_order_notes, contractor_name.strip())
                    if wo_id:
                        # Generate both Excel and HTML work orders
                        excel_data, excel_filename = download_work_order_excel(wo_id)
                        html_data, html_filename = download_work_order_html(wo_id)
                        
                        if excel_data and html_data:
                            # Show success message
                            st.success(f"✅ Work Order {wo_id} has been generated and assigned to contractor: **{contractor_name.strip()}**")
                            
                            # Display download buttons
                            st.markdown("---")
                            st.subheader("📥 Download Work Order")
                            
                            col_excel, col_html = st.columns(2)
                            
                            with col_excel:
                                st.download_button(
                                    label="📊 Download Excel Format",
                                    data=excel_data,
                                    file_name=excel_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"download_excel_{wo_id}_new",
                                    use_container_width=True
                                )
                            
                            with col_html:
                                st.download_button(
                                    label="🌐 Download HTML Format",
                                    data=html_data,
                                    file_name=html_filename,
                                    mime="text/html",
                                    key=f"download_html_{wo_id}_new",
                                    use_container_width=True
                                )
                            
                            st.info("💡 The HTML format can be opened in any web browser and is printer-friendly.")
                        
                        safe_rerun()
    
    elif current_status == "Fixed":
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ Back to In Progress", key=f"btn_back_fixed_{title}", use_container_width=True):
                update_records_status(selected_uploads, "In Progress")
                show_success_message(f"Moved {len(selected_uploads)} uploads back to In Progress")
                safe_rerun()
        
        with col2:
            if st.button("✔️ Mark as Verified", key=f"btn_verified_{title}", use_container_width=True):
                update_records_status(selected_uploads, "Verified")
                show_success_message(f"Verified {len(selected_uploads)} uploads")
                safe_rerun()

def update_records_status(upload_ids: list, new_status: str) -> int:
    """Update status for multiple uploads"""
    success_count = 0
    for upload_id in upload_ids:
        if update_repair_status(upload_id, new_status):
            success_count += 1
    return success_count

def export_status_csvs(df):
    """Export CSV files by status"""
    if 'status' not in df.columns:
        return
    status_types = df['status'].unique()
    for status in status_types:
        status_df = df[df['status'] == status]
        if not status_df.empty:
            csv = status_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {status} CSV",
                data=csv,
                file_name=f"{status.lower().replace(' ', '_')}_uploads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                key=f"download_{status.replace(' ', '_')}_csv"
            )

# -------------------------
# Work Orders Management Page - Enhanced
# -------------------------
def work_orders_page():
    """Display and manage work orders with contractor information"""
    st.header("📋 Work Orders Management")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        work_orders_df = pd.read_sql_query(
            "SELECT * FROM work_orders ORDER BY created_at DESC",
            conn
        )
        conn.close()
        
        if work_orders_df.empty:
            st.info("No work orders generated yet.")
            return
        
        # Display work orders summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Work Orders", len(work_orders_df))
        with col2:
            pending = len(work_orders_df[work_orders_df['status'] == 'Pending'])
            st.metric("Pending", pending)
        with col3:
            total_uploads = sum([len(wo_id.split(',')) for wo_id in work_orders_df['upload_ids']])
            st.metric("Total Uploads", total_uploads)
        with col4:
            total_det = work_orders_df['total_detections'].sum()
            st.metric("Total Detections", int(total_det))
        
        st.markdown("---")
        
        # Display work orders table
        st.subheader("All Work Orders")
        
        # Add filters
        status_filter = st.multiselect(
            "Filter by Status",
            options=work_orders_df['status'].unique(),
            default=work_orders_df['status'].unique()
        )
        
        filtered_wo = work_orders_df[work_orders_df['status'].isin(status_filter)]
        
        # Display each work order as an expandable card
        for idx, row in filtered_wo.iterrows():
            contractor_info = f" - Contractor: {row['contractor_name']}" if row.get('contractor_name') else ""
            with st.expander(f"🔖 {row['work_order_id']} - {row['status']} ({row['total_detections']} detections){contractor_info}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Created:** {row['created_at']}")
                    st.write(f"**Created By:** {row['created_by']}")
                    st.write(f"**Status:** {row['status']}")
                    st.write(f"**Total Uploads:** {len(row['upload_ids'].split(','))}")
                    st.write(f"**Total Detections:** {row['total_detections']}")
                
                with col2:
                    st.write(f"**Contractor:** {row.get('contractor_name', 'Not Assigned')}")
                    st.write(f"**Assigned To:** {row['assigned_to'] if row['assigned_to'] else 'Unassigned'}")
                    st.write(f"**Notes:** {row['notes'] if row['notes'] else 'No notes'}")
                
                st.markdown("---")
                
                # Show uploads in this work order
                upload_ids = row['upload_ids'].split(',')
                st.write(f"**Uploads in this Work Order ({len(upload_ids)}):**")
                
                # Get upload details
                conn = sqlite3.connect(DB_FILE)
                uploads_query = f"""
                    SELECT upload_id, location, road_name, severity, detection_count, status
                    FROM uploads 
                    WHERE upload_id IN ({','.join(['?' for _ in upload_ids])})
                """
                uploads_df = pd.read_sql_query(uploads_query, conn, params=upload_ids)
                conn.close()
                
                if not uploads_df.empty:
                    st.dataframe(uploads_df, use_container_width=True)
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Download Excel
                    excel_data, excel_filename = download_work_order_excel(row['work_order_id'])
                    if excel_data and excel_filename:
                        st.download_button(
                            label="📊 Excel",
                            data=excel_data,
                            file_name=excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_excel_{row['work_order_id']}"
                        )
                
                with col2:
                    # Download HTML
                    html_data, html_filename = download_work_order_html(row['work_order_id'])
                    if html_data and html_filename:
                        st.download_button(
                            label="🌐 HTML",
                            data=html_data,
                            file_name=html_filename,
                            mime="text/html",
                            key=f"download_html_{row['work_order_id']}"
                        )
                
                with col3:
                    if st.button("✅ Mark Complete", key=f"complete_{row['work_order_id']}"):
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE work_orders SET status = 'Completed' WHERE work_order_id = ?",
                            (row['work_order_id'],)
                        )
                        conn.commit()
                        conn.close()
                        show_success_message(f"Work Order {row['work_order_id']} marked as complete")
                        safe_rerun()
                
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{row['work_order_id']}"):
                        if st.button(f"⚠️ Confirm Delete", key=f"confirm_delete_{row['work_order_id']}"):
                            conn = sqlite3.connect(DB_FILE)
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM work_orders WHERE work_order_id = ?", (row['work_order_id'],))
                            conn.commit()
                            conn.close()
                            show_success_message(f"Work Order {row['work_order_id']} deleted")
                            safe_rerun()
        
    except Exception as e:
        st.error(f"Error loading work orders: {str(e)}")

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
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM uploads")
        cursor.execute("DELETE FROM work_orders")
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
        cursor.execute("DROP TABLE IF EXISTS detections")
        cursor.execute("DROP TABLE IF EXISTS uploads")
        cursor.execute("DROP TABLE IF EXISTS work_orders")
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
        all_uploads = get_uploads_data()
        if all_uploads.empty:
            st.warning("No data available for report generation.")
            return
        
        stats = get_repair_statistics()
        report_filename = f"FixMyStreet_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
            all_uploads.to_excel(writer, sheet_name='All_Uploads', index=False)
            
            summary_data = {
                'Metric': ['Total Uploads', 'Total Detections', 'Active Reports', 'Completed Reports'],
                'Value': [stats['total'], stats['total_detections'], stats['active'], stats['completed']]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add work orders sheet
            conn = sqlite3.connect(DB_FILE)
            work_orders_df = pd.read_sql_query("SELECT * FROM work_orders", conn)
            conn.close()
            if not work_orders_df.empty:
                work_orders_df.to_excel(writer, sheet_name='Work_Orders', index=False)
        
        with open(report_filename, 'rb') as f:
            st.download_button(
                label="📥 Download Comprehensive Report",
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
                label="📥 Download Database Backup",
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
            options=["Dashboard", "AI Detection", "Reported", "In Progress", "Fixed", "Work Orders", "Settings"],
            icons=['house', 'camera', 'exclamation-triangle', 'clock-history', 'check-circle', 'file-text', 'sliders'],
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
            st.write(f"📊 Total Uploads: {stats['total']}")
            st.write(f"🔍 Total Detections: {stats['total_detections']}")
            st.write(f"🔄 Active: {stats['active']}")
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
    
    elif selected == "Work Orders":
        work_orders_page()
    
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