# admin_interface.py
"""
Admin Interface Module for FixMyStreet AI Road Inspection System
Handles admin functionality - database management, reports, and full system control
"""

import streamlit as st
import streamlit_option_menu as option_menu
import pandas as pd
from datetime import datetime
from core_functions import *
import base64
import os
from PIL import Image
import io

def _embed_local_image_base64(path: str, crop_ratio: float = 0.7):
    """Return (img_tag, mime) for embedding local image file as base64.
    Supports PNG/JPEG/SVG. For raster images, optionally crop vertically by crop_ratio.
    """
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
        # Raster image (JPEG/PNG/etc.)
        try:
            img = Image.open(path)
        except Exception as e:
            raise
        # Optionally crop to remove whitespace/margin
        try:
            cropped = img.crop((0, 0, img.width, int(img.height * crop_ratio)))
        except Exception:
            cropped = img
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_tag = f"<img src='data:image/png;base64,{b64}' style='height:60px; margin-right:20px;'/>"
        return img_tag, 'image/png'

def move_to_fixed_database(detection_id: str) -> bool:
    """Move completed repair from active to fixed database"""
    try:
        # Read from active database
        if not os.path.exists(ACTIVE_REPAIRS_FILE):
            return False
            
        active_df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
        record_to_move = active_df[active_df['Detection_ID'] == detection_id]
        
        if record_to_move.empty:
            return False
            
        # Add completion date
        record_to_move = record_to_move.copy()
        record_to_move.loc[:, 'Date_Fixed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record_to_move.loc[:, 'Status'] = 'Fixed'
        
        # Initialize fixed database if needed
        initialize_database_file(FIXED_REPAIRS_FILE, "Fixed_Repairs")
        
        # Append to fixed database
        if os.path.exists(FIXED_REPAIRS_FILE):
            fixed_df = pd.read_excel(FIXED_REPAIRS_FILE, engine='openpyxl')
            combined_df = pd.concat([fixed_df, record_to_move], ignore_index=True)
        else:
            combined_df = record_to_move
            
        # Save to fixed database
        with pd.ExcelWriter(FIXED_REPAIRS_FILE, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Fixed_Repairs', index=False)
            apply_header_formatting(writer.sheets['Fixed_Repairs'])
        
        # Remove from active database
        remaining_df = active_df[active_df['Detection_ID'] != detection_id]
        with pd.ExcelWriter(ACTIVE_REPAIRS_FILE, engine='openpyxl') as writer:
            remaining_df.to_excel(writer, sheet_name='Road_Inspection_Data', index=False)
            apply_header_formatting(writer.sheets['Road_Inspection_Data'])
            
        return True
        
    except Exception as e:
        st.error(f"❌ Error moving record to fixed database: {str(e)}")
        return False

def update_record_status(detection_id: str, new_status: str, file_path: str = ACTIVE_REPAIRS_FILE) -> bool:
    """Update record status with automatic database migration for fixed items"""
    try:
        if not os.path.exists(file_path):
            return False
            
        df = pd.read_excel(file_path, engine='openpyxl')
        mask = df['Detection_ID'] == detection_id
        
        if not mask.any():
            return False
        
        # If promoting to 'Fixed', move the record to the fixed database
        if new_status == "Fixed":
            return move_to_fixed_database(detection_id)
        # Otherwise, just update the status in the active database
        else:
            df.loc[mask, 'Status'] = new_status
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Road_Inspection_Data', index=False)
                apply_header_formatting(writer.sheets['Road_Inspection_Data'])
            return True
    except Exception as e:
        st.error(f"❌ Error updating record status: {str(e)}")
        return False

def delete_records(selected_ids: list, file_path: str = ACTIVE_REPAIRS_FILE) -> bool:
    """Delete selected records from database"""
    try:
        if not os.path.exists(file_path):
            return False
            
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Delete associated images for deleted records
        records_to_delete = df[df['Detection_ID'].isin(selected_ids)]
        for _, record in records_to_delete.iterrows():
            try:
                img_path = record.get('Image_Path', '')
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                st.warning(f"Could not delete image for {record.get('Detection_ID', 'unknown')}: {e}")
        
        # Filter out the selected records
        df_filtered = df[~df['Detection_ID'].isin(selected_ids)]
        
        sheet_name = "Fixed_Repairs" if "fixed" in file_path.lower() else "Road_Inspection_Data"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
            apply_header_formatting(writer.sheets[sheet_name])
            
        return True
        
    except Exception as e:
        st.error(f"❌ Error deleting records: {str(e)}")
        return False

def show_inprogress_potholes_cracks(df):
    """Show only In Progress potholes and cracks in a dedicated window"""
    if df.empty or 'Status' not in df.columns or 'Defect_Type' not in df.columns:
        st.info("No In Progress potholes or cracks found.")
        return
    filtered = df[(df['Status'] == 'In Progress') & (df['Defect_Type'].isin(['Pothole', 'Crack']))]
    st.subheader("🕑 In Progress: Potholes & Cracks")
    st.dataframe(filtered)
    if not filtered.empty:
        st.download_button(
            label="📥 Download In Progress Potholes & Cracks CSV",
            data=filtered.to_csv(index=False),
            file_name="inprogress_potholes_cracks.csv",
            mime='text/csv'
        )

def admin_dashboard():
    """Admin dashboard with comprehensive statistics and in-progress window"""
    st.header("📊 Administrative Dashboard")
    
    # Load data
    active_df = pd.DataFrame()
    fixed_df = pd.DataFrame()
    
    if os.path.exists(ACTIVE_REPAIRS_FILE):
        try:
            active_df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
        except Exception as e:
            st.warning(f"Could not load active repairs: {e}")
    
    if os.path.exists(FIXED_REPAIRS_FILE):
        try:
            fixed_df = pd.read_excel(FIXED_REPAIRS_FILE, engine='openpyxl')
        except Exception as e:
            st.warning(f"Could not load fixed repairs: {e}")
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_active = len(active_df)
        st.metric("🚧 Active Repairs", total_active)
    
    with col2:
        total_fixed = len(fixed_df)
        st.metric("✅ Completed Repairs", total_fixed)
    
    with col3:
        high_priority = len(active_df[active_df['Severity'].isin(['High', 'Critical'])]) if 'Severity' in active_df.columns else 0
        st.metric("⚡ High Priority", high_priority)
    
    # Charts
    if not active_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Severity' in active_df.columns:
                st.subheader("📈 Severity Distribution")
                severity_counts = active_df['Severity'].value_counts()
                st.bar_chart(severity_counts)
        
        with col2:
            if 'Status' in active_df.columns:
                st.subheader("📊 Status Distribution")
                status_counts = active_df['Status'].value_counts()
                st.bar_chart(status_counts)
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    if not active_df.empty and 'Timestamp' in active_df.columns:
        # Sort by timestamp and show top 10
        recent_df = active_df.sort_values('Timestamp', ascending=False).head(10)
        st.dataframe(
            recent_df[['Detection_ID', 'Timestamp', 'Location', 'Defect_Type', 'Severity', 'Status', 'Inspector']]
        )
    else:
        st.info("No recent activity to display")
    
    # In Progress potholes & cracks window
    show_inprogress_potholes_cracks(active_df)

def display_database_statistics(df: pd.DataFrame, title: str):
    """Display database statistics in a nice format"""
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
        # Removed cost estimation column
        pass

def generate_work_order(selected_ids: list, df: pd.DataFrame):
    """Generate work order for selected records and embed cropped Ashoka Stambha image in HTML"""
    try:
        selected_data = df[df['Detection_ID'].isin(selected_ids)]
        # Embed local Ashoka Stambha image robustly
        try:
            img_tag, _ = _embed_local_image_base64(os.path.join('images', 'Ashoka_Piller.png'))
        except Exception:
            # Fallback: no image
            img_tag = ""
        # Work order content as HTML for browser viewing
        work_order_html = f"""
<html>
<head>
    <title>Work Order - Road Repair</title>
</head>
<body style='font-family: Arial, sans-serif;'>
    <div style='display: flex; align-items: center;'>
        {img_tag}
        <h1 style='color: #1565c0; margin: 0;'>Municipal Corporation</h1>
    </div>
    <h2 style='margin-top: 1rem;'>WORK ORDER - ROAD REPAIR</h2>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Authority:</strong> Municipal Road Department</p>
    <p><strong>Total Records:</strong> {len(selected_data)}</p>
    <hr>
    <h3>REPAIR SCHEDULE</h3>
"""
        for _, record in selected_data.iterrows():
            work_order_html += f"""
        <div style='margin-bottom: 1rem; padding: 1rem; border: 1px solid #eee; border-radius: 8px;'>
            <h4>Detection ID: {record.get('Detection_ID', 'N/A')}</h4>
            <ul>
                <li><strong>Location:</strong> {record.get('Location', 'N/A')}</li>
                <li><strong>Road:</strong> {record.get('Road_Name', 'N/A')}</li>
                <li><strong>Defect Type:</strong> {record.get('Defect_Type', 'N/A')}</li>
                <li><strong>Severity:</strong> {record.get('Severity', 'N/A')}</li>
                <li><strong>Repair Method:</strong> {record.get('Repair_Method', 'N/A')}</li>
                <li><strong>Fix Type:</strong> {record.get('Fix_Type', 'N/A')}</li>
                <li><strong>Priority Score:</strong> {record.get('Priority_Score', 'N/A')}</li>
            </ul>
        </div>
"""
        work_order_html += f"""
    <hr>
    <h3>SUMMARY</h3>
    <ul>
        <li><strong>High Priority Items:</strong> {len(selected_data[selected_data['Severity'].isin(['High', 'Critical'])])}</li>
        <li><strong>Inspector:</strong> {selected_data.iloc[0].get('Inspector', 'N/A') if not selected_data.empty else 'N/A'}</li>
    </ul>
    <p style='margin-top:2rem; color: #888;'>This work order was automatically generated by FixMyStreet AI Road Inspection System</p>
</body>
</html>
"""
        st.download_button(
            label="📥 Download Work Order (HTML)",
            data=work_order_html,
            file_name=f"work_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime='text/html'
        )
        st.markdown("<span style='color: #1565c0;'>After downloading, open the HTML file in your browser to view the cropped Ashoka Stambha and formatted work order.</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error generating work order: {str(e)}")

def export_status_csvs(df):
    """Export separate CSV files for Reported, In Progress, and Fixed repairs"""
    if 'Status' not in df.columns:
        st.warning("No 'Status' column found in data.")
        return
    status_map = {
        'Reported': 'reported_repairs.csv',
        'In Progress': 'inprogress_repairs.csv',
        'Fixed': 'fixed_repairs.csv'
    }
    for status, filename in status_map.items():
        status_df = df[df['Status'] == status]
        if not status_df.empty:
            st.download_button(
                label=f"📥 Download {status} Repairs CSV",
                data=status_df.to_csv(index=False),
                file_name=filename,
                mime='text/csv'
            )

def display_database_table(df: pd.DataFrame, title: str, file_path: str = ACTIVE_REPAIRS_FILE):
    """Display interactive database table with admin controls"""
    if df.empty:
        st.info(f"No data available in {title}")
        return
    
    st.subheader(f"📋 {title}")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Updated status filter options for Active Repairs
        if "Active" in title:
            status_options = ['Reported', 'In Progress']
        elif "Fixed" in title:
            status_options = ['Fixed']
        else:
            status_options = df['Status'].unique() if 'Status' in df.columns else []
            
        status_filter = st.multiselect(
            "Filter by Status",
            options=status_options,
            default=status_options,
            key=f"status_filter_{title.replace(' ', '_')}"
        )
    
    with col2:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=df['Severity'].unique() if 'Severity' in df.columns else [],
            default=df['Severity'].unique() if 'Severity' in df.columns else [],
            key=f"severity_filter_{title.replace(' ', '_')}"
        )
    
    with col3:
        defect_filter = st.multiselect(
            "Filter by Defect Type",
            options=df['Defect_Type'].unique() if 'Defect_Type' in df.columns else [],
            default=df['Defect_Type'].unique() if 'Defect_Type' in df.columns else [],
            key=f"defect_filter_{title.replace(' ', '_')}"
        )
    
    # Apply filters
    filtered_df = df.copy()
    if status_filter and 'Status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
    if severity_filter and 'Severity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Severity'].isin(severity_filter)]
    if defect_filter and 'Defect_Type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Defect_Type'].isin(defect_filter)]
    # Remove rows with NaN Detection_ID (cleanup)
    if 'Detection_ID' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Detection_ID'].notna()]
    st.dataframe(filtered_df)
    # Export CSVs for each status
    export_status_csvs(filtered_df)
    # Admin controls
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("🛠️ Record Management")
        
        # Record selection for actions
        selected_records = st.multiselect(
            "Select records for actions:",
            options=filtered_df['Detection_ID'].tolist(),
            key=f"select_{title.replace(' ', '_')}"
        )
        
        if selected_records:
            current_statuses = filtered_df.loc[filtered_df['Detection_ID'].isin(selected_records), 'Status'].unique()
            col1, col2, col3 = st.columns(3)
            with col1:
                # Context-aware promotion button
                if len(current_statuses) == 1:
                    current_status = current_statuses[0]
                    if current_status == 'Reported':
                        if st.button("Promote to In Progress", key=f"promote_to_inprogress_{title}"):
                            for record_id in selected_records:
                                update_record_status(record_id, "In Progress", file_path)
                            st.success("Promoted to In Progress!")
                            safe_rerun()
                    elif current_status == 'In Progress':
                        if st.button("Promote to Fixed", key=f"promote_to_fixed_{title}"):
                            for record_id in selected_records:
                                update_record_status(record_id, "Fixed", file_path)
                            st.success("Promoted to Fixed!")
                            safe_rerun()
                else:
                    st.info("Select records with the same status to promote.")
            with col2:
                if st.button("🗑️ Delete Selected", key=f"delete_{title}"):
                    if delete_records(selected_records, file_path):
                        st.success(f"Deleted {len(selected_records)} records.")
                        safe_rerun()
            with col3:
                if st.button("📧 Generate Work Order", key=f"workorder_{title}"):
                    generate_work_order(selected_records, filtered_df)

def admin_controls():
    """Display admin control panel"""
    st.subheader("🔧 Admin Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if safe_button("🗑️ Clear Active DB", use_container_width=True):
            if os.path.exists(ACTIVE_REPAIRS_FILE):
                os.remove(ACTIVE_REPAIRS_FILE)
                show_success_message("Active database cleared successfully!")
                safe_rerun()
    
    with col2:
        if safe_button("🗑️ Clear Fixed DB", use_container_width=True):
            if os.path.exists(FIXED_REPAIRS_FILE):
                os.remove(FIXED_REPAIRS_FILE)
                show_success_message("Fixed database cleared successfully!")
                safe_rerun()
    
    with col3:
        if safe_button("📊 Generate Report", use_container_width=True):
            generate_comprehensive_report()

def generate_comprehensive_report():
    """Generate comprehensive repair report"""
    try:
        # Load both databases
        active_data = []
        fixed_data = []
        
        if os.path.exists(ACTIVE_REPAIRS_FILE):
            active_df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
            active_data = active_df.to_dict('records')
        
        if os.path.exists(FIXED_REPAIRS_FILE):
            fixed_df = pd.read_excel(FIXED_REPAIRS_FILE, engine='openpyxl')
            fixed_data = fixed_df.to_dict('records')
        
        # Generate report filename
        report_filename = f"FixMyStreet_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
            # Active repairs sheet
            if active_data:
                active_df.to_excel(writer, sheet_name='Active_Repairs', index=False)
                apply_header_formatting(writer.sheets['Active_Repairs'])
            
            # Fixed repairs sheet
            if fixed_data:
                fixed_df.to_excel(writer, sheet_name='Fixed_Repairs', index=False)
                apply_header_formatting(writer.sheets['Fixed_Repairs'])
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Active Repairs', 'Total Fixed Repairs'],
                'Value': [
                    len(active_data),
                    len(fixed_data)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            apply_header_formatting(writer.sheets['Summary'])
        
        st.success(f"✅ Report generated: {report_filename}")
        
        # Offer download
        with open(report_filename, 'rb') as f:
            st.download_button(
                label="📥 Download Report",
                data=f.read(),
                file_name=report_filename,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")

def admin_panel():
    """Administrative panel with system management tools"""
    st.header("⚙️ Administrative Panel")
    
    # System information
    st.subheader("🖥️ System Information")
    col1, col2, col3 = st.columns(3)
    
    model, _ = load_yolo_model()
    geolocator = initialize_geocoder()
    
    with col1:
        st.info(f"**Model Status:** {'✅ Loaded' if model else '❌ Not Available'}")
    
    with col2:
        # Removed geocoding status display entirely
        pass
    
    with col3:
        st.info(f"**Active DB:** {'✅ Exists' if os.path.exists(ACTIVE_REPAIRS_FILE) else '❌ Missing'}")
    
    # Admin controls
    admin_controls()
    
    # Database maintenance
    st.subheader("🔧 Database Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Initialize Databases:**")
        if safe_button("🔄 Initialize Active DB"):
            if initialize_database_file(ACTIVE_REPAIRS_FILE):
                show_success_message("Active database initialized!")
                safe_rerun()
        
        if safe_button("🔄 Initialize Fixed DB"):
            if initialize_database_file(FIXED_REPAIRS_FILE, "Fixed_Repairs"):
                show_success_message("Fixed database initialized!")
                safe_rerun()
    
    with col2:
        st.markdown("**Database Information:**")
        if os.path.exists(ACTIVE_REPAIRS_FILE):
            size = os.path.getsize(ACTIVE_REPAIRS_FILE)
            st.text(f"Active DB Size: {size:,} bytes")
        
        if os.path.exists(FIXED_REPAIRS_FILE):
            size = os.path.getsize(FIXED_REPAIRS_FILE)
            st.text(f"Fixed DB Size: {size:,} bytes")

def admin_main():
    """Main function for admin interface"""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.show_success = False
        st.session_state.success_message = ""
    display_session_messages()
    # Check authentication
    if not st.session_state.logged_in or st.session_state.user_role != "admin":
        login_interface()
        return
    # Load model and geocoder
    model, model_status = load_yolo_model()
    geolocator = initialize_geocoder()
    if model_status and model is None:
        st.info(model_status)
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'><h3 style='color: white; margin: 0;'>Welcome!</h3><p style='color: white; margin: 0;'>{st.session_state.user_name}</p><small style='color: #e0e0e0;'>Role: Administrator</small></div>", unsafe_allow_html=True)
        selected = option_menu.option_menu(
            menu_title="Navigation",
            options=["🏠 Dashboard", "🔍 AI Detection", "📋 Active Repairs", "🕑 In Progress Work", "✅ Fixed Repairs", "⚙️ Admin Panel"],
            icons=['house', 'camera', 'list-task', 'clock-history', 'check-circle', 'gear'],
            menu_icon="cast",
            default_index=0,
        )
        st.markdown("---")
        if safe_button("🚪 Logout"):
            logout()
    # Main content based on navigation
    if selected == "🏠 Dashboard":
        admin_dashboard()
    elif selected == "🔍 AI Detection":
        detection_interface(model, geolocator)
    elif selected == "📋 Active Repairs":
        st.header("🚧 Active Repairs Management")
        if os.path.exists(ACTIVE_REPAIRS_FILE):
            try:
                active_df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
                display_database_statistics(active_df, "Active Repairs Database")
                display_database_table(
                    active_df,
                    "Active Repairs",
                    file_path=ACTIVE_REPAIRS_FILE
                )
            except Exception as e:
                st.error(f"❌ Error loading active repairs: {str(e)}")
        else:
            st.info("📝 No active repairs database found. Start by detecting some defects!")
    elif selected == "🕑 In Progress Work":
        st.header("🕑 In Progress Work")
        if os.path.exists(ACTIVE_REPAIRS_FILE):
            try:
                active_df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
                inprogress_df = active_df[active_df['Status'] == 'In Progress']
                display_database_statistics(inprogress_df, "In Progress Repairs")
                display_database_table(
                    inprogress_df,
                    "In Progress Repairs",
                    file_path=ACTIVE_REPAIRS_FILE
                )
            except Exception as e:
                st.error(f"❌ Error loading in progress repairs: {str(e)}")
        else:
            st.info("📝 No in progress repairs found.")
    elif selected == "✅ Fixed Repairs":
        st.header("✅ Fixed Repairs Archive")
        if os.path.exists(FIXED_REPAIRS_FILE):
            try:
                fixed_df = pd.read_excel(FIXED_REPAIRS_FILE, engine='openpyxl')
                display_database_statistics(fixed_df, "Fixed Repairs Database")
                display_database_table(
                    fixed_df,
                    "Fixed Repairs",
                    file_path=FIXED_REPAIRS_FILE
                )
            except Exception as e:
                st.error(f"❌ Error loading fixed repairs: {str(e)}")
        else:
            st.info("📝 No fixed repairs database found.")
    elif selected == "⚙️ Admin Panel":
        admin_panel()