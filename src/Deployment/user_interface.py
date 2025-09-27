"""
User Interface Module for FixMyStreet AI Road Inspection System
CORRECTED for upload-based system with proper imports
"""

import os
import tempfile
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

import streamlit as st
import streamlit_option_menu as option_menu
from core_functions import (
    authenticate_user, register_user, initialize_database, 
    show_success_message, safe_rerun, display_session_messages,
    load_yolo_model, initialize_geocoder,
    detection_interface, initialize_app, logout, DB_FILE,
    get_uploads_data, get_repair_statistics, login_interface,
    get_detections_for_upload
)

# -------------------------
# User Dashboard Functions
# -------------------------
def user_dashboard():
    """Enhanced dashboard for users showing their upload activity"""
    st.header("Inspector Dashboard")
    
    # Load user's upload history from SQLite database
    user_uploads_count = 0
    user_detections_count = 0
    recent_uploads = pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get user's upload count
        query_count = "SELECT COUNT(*) as count FROM uploads WHERE inspector = ?"
        count_result = pd.read_sql_query(query_count, conn, params=(st.session_state.user_name,))
        user_uploads_count = count_result['count'].iloc[0] if not count_result.empty else 0
        
        # Get user's total detections count
        query_detections = """
        SELECT SUM(detection_count) as count 
        FROM uploads
        WHERE inspector = ?
        """
        detections_result = pd.read_sql_query(query_detections, conn, params=(st.session_state.user_name,))
        user_detections_count = int(detections_result['count'].iloc[0]) if not detections_result.empty and detections_result['count'].iloc[0] else 0
        
        # Get recent uploads (last 10) with defect types
        query_recent = """
        SELECT u.upload_id, u.timestamp, u.location, u.road_name, 
               u.severity, u.status, u.detection_count,
               GROUP_CONCAT(DISTINCT d.defect_type) as defect_types,
               u.gps_lat, u.gps_lon
        FROM uploads u
        LEFT JOIN detections d ON u.upload_id = d.upload_id
        WHERE u.inspector = ? 
        GROUP BY u.upload_id
        ORDER BY u.timestamp DESC 
        LIMIT 10
        """
        recent_uploads = pd.read_sql_query(query_recent, conn, params=(st.session_state.user_name,))
        
        conn.close()
        
    except sqlite3.Error as e:
        st.warning(f"Could not load upload history: {e}")
        if 'conn' in locals():
            conn.close()
    except Exception as e:
        st.warning(f"Unexpected error loading data: {e}")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Uploads", user_uploads_count)
    
    with col2:
        st.metric("Total Defects Found", user_detections_count)
    
    with col3:
        today_count = 0
        if not recent_uploads.empty and 'timestamp' in recent_uploads.columns:
            today = datetime.now().date()
            try:
                recent_uploads['timestamp'] = pd.to_datetime(recent_uploads['timestamp'])
                today_count = len(recent_uploads[
                    recent_uploads['timestamp'].dt.date == today
                ])
            except Exception:
                today_count = 0
        st.metric("Today's Uploads", today_count)
    
    with col4:
        high_severity = 0
        if not recent_uploads.empty and 'severity' in recent_uploads.columns:
            high_severity = len(recent_uploads[
                recent_uploads['severity'].isin(['High', 'Critical'])
            ])
        st.metric("High Priority", high_severity)
    
    # Recent uploads table
    st.subheader("Your Recent Uploads")
    if not recent_uploads.empty:
        # Format the display dataframe
        display_df = recent_uploads.copy()
        
        # Format timestamp for better display
        if 'timestamp' in display_df.columns:
            try:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                pass
        
        # Select and order columns for display
        display_columns = ['upload_id', 'timestamp', 'location', 'road_name', 
                         'detection_count', 'defect_types', 'severity', 'status']
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        # Display the dataframe
        st.dataframe(
            display_df[available_columns], 
            hide_index=True,
            use_container_width=True
        )
        
        # Show details of selected upload
        selected_upload = st.selectbox(
            "Select an upload to view details:",
            options=recent_uploads['upload_id'].tolist(),
            key="select_upload_details"
        )
        
        if selected_upload:
            show_upload_details(selected_upload)
        
        # Download option
        csv = recent_uploads.to_csv(index=False)
        st.download_button(
            label="📥 Download Your Uploads as CSV",
            data=csv,
            file_name=f"my_uploads_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export all user data
        if st.button("📊 Export All My Data"):
            export_all_user_data()
    
    else:
        st.info("You haven't submitted any inspection reports yet. Use the Detection feature to start!")
        
        # Add helpful tips for new users
        with st.expander("💡 Getting Started Tips"):
            st.markdown("""
            **How to start detecting road defects:**
            1. Go to **AI Detection** from the sidebar
            2. Upload images or videos of roads
            3. Let our AI analyze and detect issues
            4. Add location information
            5. Submit your findings
            
            **Best practices:**
            - Use clear, well-lit images
            - Include multiple angles of defects
            - Add accurate location details
            - Review AI suggestions before submitting
            
            **What happens after submission:**
            - Each upload gets a unique ID
            - Multiple defects can be detected in one upload
            - Admins can generate work orders based on your uploads
            """)

def show_upload_details(upload_id: str):
    """Show detailed information about a specific upload"""
    with st.expander(f"📋 Details for {upload_id}", expanded=True):
        try:
            conn = sqlite3.connect(DB_FILE)
            
            # Get upload details
            upload_query = "SELECT * FROM uploads WHERE upload_id = ?"
            upload_df = pd.read_sql_query(upload_query, conn, params=(upload_id,))
            
            if not upload_df.empty:
                upload = upload_df.iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Upload Information:**")
                    st.write(f"📍 Location: {upload['location']}")
                    st.write(f"🛣️ Road: {upload['road_name']}")
                    st.write(f"📅 Date: {upload['timestamp']}")
                    st.write(f"⚠️ Severity: {upload['severity']}")
                    st.write(f"📊 Status: {upload['status']}")
                
                with col2:
                    st.markdown("**Detection Summary:**")
                    st.write(f"🔍 Total Defects: {upload['detection_count']}")
                    if upload['work_order_id']:
                        st.write(f"📄 Work Order: {upload['work_order_id']}")
                
                if upload['notes']:
                    st.markdown("**Notes:**")
                    st.write(upload['notes'])
                
                # Get and display detections
                detections_query = "SELECT * FROM detections WHERE upload_id = ?"
                detections_df = pd.read_sql_query(detections_query, conn, params=(upload_id,))
                
                if not detections_df.empty:
                    st.markdown(f"**Detected Defects ({len(detections_df)}):**")
                    display_cols = ['detection_id', 'defect_type', 'confidence', 'repair_method', 'priority_score']
                    available_cols = [col for col in display_cols if col in detections_df.columns]
                    st.dataframe(
                        detections_df[available_cols],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Download detections
                    csv = detections_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detections CSV",
                        data=csv,
                        file_name=f"{upload_id}_detections.csv",
                        mime="text/csv",
                        key=f"download_{upload_id}"
                    )
                else:
                    st.info("No detections found for this upload.")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error loading upload details: {e}")

def export_all_user_data():
    """Export all user data including uploads and detections"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get all user uploads
        uploads_query = """
        SELECT * FROM uploads 
        WHERE inspector = ? 
        ORDER BY timestamp DESC
        """
        uploads_df = pd.read_sql_query(uploads_query, conn, params=(st.session_state.user_name,))
        
        # Get all detections for user's uploads
        detections_query = """
        SELECT d.* FROM detections d
        JOIN uploads u ON d.upload_id = u.upload_id
        WHERE u.inspector = ?
        ORDER BY d.upload_id, d.detection_id
        """
        detections_df = pd.read_sql_query(detections_query, conn, params=(st.session_state.user_name,))
        
        conn.close()
        
        # Create Excel file
        import io
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            uploads_df.to_excel(writer, sheet_name='My_Uploads', index=False)
            detections_df.to_excel(writer, sheet_name='My_Detections', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Uploads', 'Total Detections', 'Export Date', 'Inspector'],
                'Value': [len(uploads_df), len(detections_df), datetime.now().strftime('%Y-%m-%d'), st.session_state.user_name]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        
        # Offer download
        filename = f"my_complete_data_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        st.download_button(
            label="📥 Download Complete Export",
            data=output,
            file_name=filename,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        st.success("✅ Complete export ready for download!")
        
    except Exception as e:
        st.error(f"Error exporting data: {e}")

def load_user_statistics():
    """Load additional statistics for the user dashboard"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get statistics by defect type
        defect_query = """
        SELECT d.defect_type, COUNT(*) as count, AVG(d.priority_score) as avg_priority
        FROM detections d
        JOIN uploads u ON d.upload_id = u.upload_id
        WHERE u.inspector = ? 
        GROUP BY d.defect_type
        ORDER BY count DESC
        """
        defect_stats = pd.read_sql_query(defect_query, conn, params=(st.session_state.user_name,))
        
        # Get monthly activity
        monthly_query = """
        SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as uploads
        FROM uploads 
        WHERE inspector = ? 
        GROUP BY strftime('%Y-%m', timestamp)
        ORDER BY month DESC
        LIMIT 12
        """
        monthly_stats = pd.read_sql_query(monthly_query, conn, params=(st.session_state.user_name,))
        
        conn.close()
        return defect_stats, monthly_stats
        
    except Exception as e:
        st.warning(f"Could not load additional statistics: {e}")
        return pd.DataFrame(), pd.DataFrame()

def show_detailed_statistics():
    """Show detailed statistics in an expandable section"""
    with st.expander("📊 Detailed Statistics"):
        defect_stats, monthly_stats = load_user_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Defects by Type")
            if not defect_stats.empty:
                st.dataframe(defect_stats, hide_index=True, use_container_width=True)
            else:
                st.info("No defect data available")
        
        with col2:
            st.subheader("Monthly Activity")
            if not monthly_stats.empty:
                st.bar_chart(monthly_stats.set_index('month')['uploads'])
            else:
                st.info("No monthly data available")

# -------------------------
# Navigation and Main Interface
# -------------------------
def create_sidebar_navigation():
    """Create enhanced sidebar navigation for users"""
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='color: white; margin: 0;'>Welcome!</h3>
            <p style='color: white; margin: 0; font-weight: bold;'>{st.session_state.user_name}</p>
            <small style='color: #e0e0e0;'>Role: Inspector</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        menu_options = ["Dashboard", "AI Detection", "Settings"]
        menu_icons = ['house', 'camera', 'sliders']
        
        selected = option_menu.option_menu(
            menu_title="Navigation",
            options=menu_options,
            icons=menu_icons,
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        try:
            conn = sqlite3.connect(DB_FILE)
            total_query = "SELECT COUNT(*) as total_count FROM uploads WHERE inspector = ?"
            detections_query = """
            SELECT SUM(detection_count) as det_count FROM uploads
            WHERE inspector = ?
            """
            
            total_result = pd.read_sql_query(total_query, conn, params=(st.session_state.user_name,))
            det_result = pd.read_sql_query(detections_query, conn, params=(st.session_state.user_name,))
            
            total_count = total_result['total_count'].iloc[0] if not total_result.empty else 0
            det_count = int(det_result['det_count'].iloc[0]) if not det_result.empty and det_result['det_count'].iloc[0] else 0
            conn.close()
            
            st.markdown("### Quick Stats")
            st.markdown(f"**Total Uploads:** {total_count}")
            st.markdown(f"**Total Defects:** {det_count}")
            
        except Exception:
            pass
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", key="logout_btn"):
            logout()
        
        return selected

# -------------------------
# Main Application Function
# -------------------------
def user_main():
    """Main function for user interface"""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.username = None
        st.session_state.user_id = None
        st.session_state.show_success = False
        st.session_state.success_message = ""
    
    # Initialize database and app
    if not initialize_app():
        st.error("Failed to initialize application. Please check your setup.")
        return
    
    # Display session messages
    display_session_messages()
    
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        login_interface()
        return
    
    # Check if user is an inspector (prevent admin access)
    if st.session_state.user_role != 'inspector':
        st.error("This interface is for inspectors only. Please use the admin interface.")
        if st.button("Logout"):
            logout()
        return
    
    # Load model and geocoder
    model, model_status = load_yolo_model()
    geolocator = initialize_geocoder()
    
    # Create sidebar navigation
    selected = create_sidebar_navigation()
    
    # Display model status in sidebar
    st.sidebar.info(model_status)
    
    # Main content area based on navigation selection
    if selected == "Dashboard":
        user_dashboard()
        show_detailed_statistics()
    
    elif selected == "AI Detection":
        # Check if model is available before showing detection interface
        if model is None:
            st.error("AI model is not loaded. Cannot perform detection.")
            st.info("Please contact the administrator to resolve this issue.")
            return
        
        detection_interface(model, geolocator)
    
    elif selected == "Settings":
        st.header("⚙️ Settings")
        st.subheader("User Information")
        st.write(f"**Name:** {st.session_state.user_name}")
        st.write(f"**Username:** {st.session_state.username}")
        st.write(f"**Role:** {st.session_state.user_role}")
        
        st.subheader("System Information")
        st.write(f"**Database:** SQLite ({DB_FILE})")
        st.write(f"**Model Status:** {model_status}")
        
        # Add user preferences section
        st.subheader("Preferences")
        
        # Theme preference (placeholder for future implementation)
        theme_pref = st.selectbox("Preferred Theme", ["Default", "Dark", "Light"])
        
        # Notification preferences
        email_notifications = st.checkbox("Email Notifications", value=True)
        
        # Default location for detections
        default_location = st.text_input("Default Location", placeholder="e.g., City, State")
        
        # Save preferences (placeholder)
        if st.button("Save Preferences"):
            st.success("Preferences saved successfully!")
        
        # Account statistics
        st.markdown("---")
        st.subheader("📊 Account Statistics")
        try:
            conn = sqlite3.connect(DB_FILE)
            
            # Get user statistics
            stats_query = """
            SELECT 
                COUNT(DISTINCT u.upload_id) as total_uploads,
                SUM(u.detection_count) as total_detections,
                MIN(u.timestamp) as first_upload,
                MAX(u.timestamp) as last_upload
            FROM uploads u
            WHERE u.inspector = ?
            """
            stats_df = pd.read_sql_query(stats_query, conn, params=(st.session_state.user_name,))
            conn.close()
            
            if not stats_df.empty:
                stats = stats_df.iloc[0]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Uploads", int(stats['total_uploads']) if stats['total_uploads'] else 0)
                    if stats['first_upload']:
                        st.write(f"**First Upload:** {stats['first_upload']}")
                
                with col2:
                    st.metric("Total Detections", int(stats['total_detections']) if stats['total_detections'] else 0)
                    if stats['last_upload']:
                        st.write(f"**Last Upload:** {stats['last_upload']}")
        
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="FixMyStreet - Inspector Interface",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run main app
    user_main()