# user_interface.py
"""
User Interface Module for FixMyStreet AI Road Inspection System
Handles regular user/inspector functionality - detection only
"""

import streamlit as st
import streamlit_option_menu as option_menu
from core_functions import *

def user_dashboard():
    """Simple dashboard for users showing their detection activity"""
    st.header("👷‍♂️ Inspector Dashboard")
    
    # Load user's detection history (last 10 detections)
    user_detections = 0
    recent_detections = []
    
    if os.path.exists(ACTIVE_REPAIRS_FILE):
        try:
            df = pd.read_excel(ACTIVE_REPAIRS_FILE, engine='openpyxl')
            user_df = df[df['Inspector'] == st.session_state.user_name]
            user_detections = len(user_df)
            recent_detections = user_df.sort_values('Timestamp', ascending=False).head(10)
        except Exception as e:
            st.warning(f"Could not load detection history: {e}")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Detections", user_detections)
    
    with col2:
        today_count = 0
        if not recent_detections.empty and 'Timestamp' in recent_detections.columns:
            today = datetime.now().date()
            today_count = len(recent_detections[
                pd.to_datetime(recent_detections['Timestamp']).dt.date == today
            ])
        st.metric("📅 Today's Detections", today_count)
    
    with col3:
        high_severity = 0
        if not recent_detections.empty and 'Severity' in recent_detections.columns:
            high_severity = len(recent_detections[
                recent_detections['Severity'].isin(['High', 'Critical'])
            ])
        st.metric("⚠️ High Priority Found", high_severity)
    
    with col4:
        st.metric("👤 Inspector", st.session_state.user_name)
    
    # Recent detections
    st.subheader("🕒 Your Recent Detections")
    if not recent_detections.empty:
        # Show only relevant columns for user
        display_columns = ['Detection_ID', 'Timestamp', 'Location', 'Road_Name', 
                         'Defect_Type', 'Severity', 'Status']
        available_columns = [col for col in display_columns if col in recent_detections.columns]
        st.dataframe(recent_detections[available_columns], hide_index=True)
    else:
        st.info("No detections found. Start by uploading images or videos for analysis!")

def user_main():
    """Main function for user interface"""
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.show_success = False
        st.session_state.success_message = ""
    display_session_messages()
    # Check authentication
    if not st.session_state.logged_in or st.session_state.user_role != "inspector":
        login_interface()
        return
    # Load model and geocoder
    model, model_status = load_yolo_model()
    geolocator = initialize_geocoder()
    if model_status and model is None:
        st.info(model_status)
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'><h3 style='color: white; margin: 0;'>Welcome!</h3><p style='color: white; margin: 0;'>{st.session_state.user_name}</p><small style='color: #e0e0e0;'>Role: Inspector</small></div>", unsafe_allow_html=True)
        selected = option_menu.option_menu(
            menu_title="Navigation",
            options=["🏠 Dashboard", "🔍 AI Detection"],
            icons=['house', 'camera'],
            menu_icon="cast",
            default_index=0,
        )
        st.markdown("---")
        if safe_button("🚪 Logout"):
            logout()
    # Main content based on navigation
    if selected == "🏠 Dashboard":
        user_dashboard()
    elif selected == "🔍 AI Detection":
        detection_interface(model, geolocator)