"""
User Interface Module for FixMyStreet AI Road Inspection System
Handles regular user/inspector functionality - detection and dashboard only
Updated to use SQLite database with OTP verification for registration
"""

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
    get_repairs_data, get_repair_statistics, login_interface
)

# -------------------------
# User Dashboard Functions
# -------------------------
def user_dashboard():
    """Enhanced dashboard for users showing their detection activity"""
    st.header("Inspector Dashboard")
    
    # Load user's detection history from SQLite database
    user_detections = 0
    recent_detections = pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get user's detection count
        query_count = "SELECT COUNT(*) as count FROM repairs WHERE inspector = ?"
        count_result = pd.read_sql_query(query_count, conn, params=(st.session_state.user_name,))
        user_detections = count_result['count'].iloc[0] if not count_result.empty else 0
        
        # Get recent detections (last 10)
        query_recent = """
        SELECT detection_id, timestamp, location, road_name, defect_type, 
               severity, status, priority_score, gps_lat, gps_lon
        FROM repairs 
        WHERE inspector = ? 
        ORDER BY timestamp DESC 
        LIMIT 10
        """
        recent_detections = pd.read_sql_query(query_recent, conn, params=(st.session_state.user_name,))
        
        conn.close()
        
    except sqlite3.Error as e:
        st.warning(f"Could not load detection history: {e}")
        if 'conn' in locals():
            conn.close()
    except Exception as e:
        st.warning(f"Unexpected error loading data: {e}")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", user_detections)
    
    with col2:
        today_count = 0
        if not recent_detections.empty and 'timestamp' in recent_detections.columns:
            today = datetime.now().date()
            try:
                recent_detections['timestamp'] = pd.to_datetime(recent_detections['timestamp'])
                today_count = len(recent_detections[
                    recent_detections['timestamp'].dt.date == today
                ])
            except Exception:
                today_count = 0
        st.metric("Today's Detections", today_count)
    
    with col3:
        high_severity = 0
        if not recent_detections.empty and 'severity' in recent_detections.columns:
            high_severity = len(recent_detections[
                recent_detections['severity'].isin(['High', 'Critical'])
            ])
        st.metric("High Priority Found", high_severity)
    
    with col4:
        st.metric("Inspector", st.session_state.user_name)
    
    # Recent detections table
    st.subheader("Your Recent Detections")
    if not recent_detections.empty:
        # Format the display dataframe
        display_df = recent_detections.copy()
        
        # Format timestamp for better display
        if 'timestamp' in display_df.columns:
            try:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                pass  # Keep original format if conversion fails
        
        # Select and order columns for display
        display_columns = ['detection_id', 'timestamp', 'location', 'road_name', 
                         'defect_type', 'severity', 'status']
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        # Display the dataframe
        st.dataframe(
            display_df[available_columns], 
            hide_index=True,
            use_container_width=True
        )
        
        # Download option
        csv = recent_detections.to_csv(index=False)
        st.download_button(
            label="Download Your Reports as CSV",
            data=csv,
            file_name=f"my_reports_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Add export option for all user data
        if st.button("Export All My Detections"):
            try:
                conn = sqlite3.connect(DB_FILE)
                export_query = """
                SELECT * FROM repairs 
                WHERE inspector = ? 
                ORDER BY timestamp DESC
                """
                export_df = pd.read_sql_query(export_query, conn, params=(st.session_state.user_name,))
                conn.close()
                
                # Convert to CSV for download
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Complete History CSV",
                    data=csv,
                    file_name=f"complete_detections_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                st.success("Complete export ready for download!")
                
            except Exception as e:
                st.error(f"Error exporting data: {e}")
    else:
        st.info("You haven't submitted any inspection reports yet. Use the Detection feature to start!")
        
        # Add helpful tips for new users
        with st.expander("Getting Started Tips"):
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
            """)

def load_user_statistics():
    """Load additional statistics for the user dashboard"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get statistics by defect type
        defect_query = """
        SELECT defect_type, COUNT(*) as count, AVG(priority_score) as avg_priority
        FROM repairs 
        WHERE inspector = ? 
        GROUP BY defect_type
        ORDER BY count DESC
        """
        defect_stats = pd.read_sql_query(defect_query, conn, params=(st.session_state.user_name,))
        
        # Get monthly activity
        monthly_query = """
        SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as detections
        FROM repairs 
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
    with st.expander("Detailed Statistics"):
        defect_stats, monthly_stats = load_user_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Defects by Type")
            if not defect_stats.empty:
                st.dataframe(defect_stats, hide_index=True)
            else:
                st.info("No defect data available")
        
        with col2:
            st.subheader("Monthly Activity")
            if not monthly_stats.empty:
                st.bar_chart(monthly_stats.set_index('month')['detections'])
            else:
                st.info("No monthly data available")

# -------------------------
# Navigation and Main Interface
# -------------------------
def create_sidebar_navigation():
    """Create enhanced sidebar navigation for users"""
    with st.sidebar:
        # User info section with enhanced styling
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='color: white; margin: 0;'>Welcome!</h3>
            <p style='color: white; margin: 0; font-weight: bold;'>{st.session_state.user_name}</p>
            <small style='color: #e0e0e0;'>Role: Inspector</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu for users
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
            # Users see their own stats
            total_query = "SELECT COUNT(*) as total_count FROM repairs WHERE inspector = ?"
            today_query = "SELECT COUNT(*) as today_count FROM repairs WHERE inspector = ? AND DATE(timestamp) = DATE('now')"
            
            total_result = pd.read_sql_query(total_query, conn, params=(st.session_state.user_name,))
            today_result = pd.read_sql_query(today_query, conn, params=(st.session_state.user_name,))
            
            total_count = total_result['total_count'].iloc[0] if not total_result.empty else 0
            today_count = today_result['today_count'].iloc[0] if not today_result.empty else 0
            conn.close()
            
            st.markdown("### Quick Stats")
            st.markdown(f"**Total Reports:** {total_count}")
            st.markdown(f"**Today's Reports:** {today_count}")
            
        except Exception:
            pass  # Silently handle errors in sidebar stats
        
        st.markdown("---")
        
        # Logout button
        if st.button("Logout", key="logout_btn"):
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
        st.header("Settings")
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

# -------------------------
# Helper Functions
# -------------------------
def safe_button(label, key=None):
    """Safe button wrapper to prevent rerun issues"""
    return st.button(label, key=key)

def execute_db_query(query, params=None, fetch=False):
    """Execute database query with proper error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            result = cursor.fetchall()
            conn.close()
            return result
        else:
            conn.commit()
            conn.close()
            return True
            
    except sqlite3.Error as e:
        if conn:
            conn.close()
        raise e

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