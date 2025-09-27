#!/usr/bin/env python3
"""
FixMyStreet AI Road Inspection System - Main Application
A comprehensive AI-powered road defect detection and management system

Run with: streamlit run app.py

Author: AI Road Inspection Team
Version: 2.0
"""

import os
import sys
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="FixMyStreet - AI Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/fixmystreet/ai-inspection',
        'Report a bug': 'https://github.com/fixmystreet/ai-inspection/issues',
        'About': """
        # FixMyStreet AI Road Inspection System
        
        An intelligent system for automated road defect detection and management.
        
        **Features:**
        - AI-powered defect detection using YOLOv8
        - Real-time image and video processing
        - Comprehensive database management
        - Multi-user access control
        
        Built with Streamlit, OpenCV, and Ultralytics YOLO.
        """
    }
)

# Import custom modules with error handling
try:
    from core_functions import (
        load_yolo_model, initialize_geocoder, login_interface, logout,
        detection_interface, display_session_messages, safe_rerun,
        initialize_app
    )
    from admin_interface import admin_main
    from user_interface import user_main
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all required files are present:")
    st.error("- core_functions.py")
    st.error("- admin_interface.py") 
    st.error("- user_interface.py")
    st.stop()

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    required_packages = [
        'cv2', 'numpy', 'pandas', 'openpyxl', 'geopy', 
        'ultralytics', 'PIL', 'streamlit_option_menu'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'streamlit_option_menu':
                import streamlit_option_menu
            else:
                __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    return missing_deps

def display_system_status():
    """Display system status and diagnostics"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    # Check model availability
    model, model_status = load_yolo_model()
    if model:
        st.sidebar.success("YOLO Model: Ready")
    else:
        st.sidebar.error("YOLO Model: Missing")
        st.sidebar.caption("Place 'best.pt' in app directory")
    
    # Check database
    from core_functions import DB_FILE
    
    if os.path.exists(DB_FILE):
        st.sidebar.success("Database: Connected")
    else:
        st.sidebar.info("Database: Will be created")

def show_welcome_screen():
    """Display welcome screen with system information"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #2c3e50; margin-bottom: 0.5rem; font-size: 3.5rem;'>🛣️ FixMyStreet</h1>
        <h2 style='color: #7f8c8d; font-weight: 300; margin-bottom: 2rem;'>AI-Powered Road Inspection System</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
        <h3 style='margin-top: 0;'>Advanced Road Infrastructure Management</h3>
        <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>Leverage cutting-edge AI technology to detect, analyze, and manage road defects with unprecedented accuracy and efficiency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key Features")
    
    features = [
        ("AI Detection", "YOLOv8-powered real-time defect identification"),
        ("Smart Analytics", "Comprehensive cost estimation and priority scoring"),
        ("Database Management", "Robust SQLite-based record keeping system"),
        ("Role-Based Access", "Separate interfaces for inspectors and administrators"),
        ("Cost Optimization", "Accurate repair cost estimates based on Indian standards")
    ]
    
    for title, desc in features:
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #667eea; border-radius: 5px;'>
            <strong style='color: #2c3e50;'>{title}</strong><br>
            <span style='color: #6c757d;'>{desc}</span>
        </div>
        """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'logged_in': False,
        'user_role': None,
        'user_name': None,
        'username': None,
        'user_id': None,
        'show_success': False,
        'success_message': "",
        'app_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_error_page(error_msg: str):
    """Display error page with troubleshooting information"""
    st.error("Application Error")
    st.markdown(f"""
    **Error Details:**
    ```
    {error_msg}
    ```
    
    **Troubleshooting Steps:**
    
    1. **Check File Structure:**
       - Ensure all Python files are in the same directory
       - Verify `best.pt` model file is present (optional but recommended)
    
    2. **Install Dependencies:**
       ```bash
       pip install streamlit opencv-python numpy pandas openpyxl
       pip install geopy ultralytics pillow streamlit-option-menu
       ```
    
    3. **Restart Application:**
       ```bash
       streamlit run app.py
       ```
    
    4. **Contact Support:**
       - Check GitHub repository for updates
       - Report issues with full error details
    """)

def main():
    """Main application entry point"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Check dependencies
        missing_deps = check_dependencies()
        if missing_deps:
            st.error("Missing Required Dependencies")
            st.markdown("**Please install the following packages:**")
            for dep in missing_deps:
                st.code(f"pip install {dep}")
            st.markdown("**Installation command:**")
            st.code("pip install " + " ".join(missing_deps))
            st.stop()
        
        # Initialize the application
        if not initialize_app():
            st.error("Failed to initialize application. Please check your setup.")
            return
        
        # Display session messages
        display_session_messages()
        
        # Show system status in sidebar
        display_system_status()
        
        # Main application logic
        if not st.session_state.logged_in:
            # Show welcome screen and login
            show_welcome_screen()
            st.markdown("---")
            login_interface()
            return
        
        # Route to appropriate interface based on user role
        try:
            if st.session_state.user_role == "admin":
                admin_main()
            elif st.session_state.user_role == "inspector":
                user_main()
            else:
                st.error("Invalid user role. Please contact administrator.")
                if st.button("Logout"):
                    logout()
                    
        except Exception as interface_error:
            st.error(f"Interface Error: {str(interface_error)}")
            st.markdown("**Try refreshing the page or logging out and back in.**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refresh Page"):
                    safe_rerun()
            with col2:
                if st.button("Logout"):
                    logout()
        
    except Exception as e:
        show_error_page(str(e))
        
        # Emergency logout option
        if st.button("Emergency Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            safe_rerun()

def run_diagnostics():
    """Run system diagnostics and display results"""
    st.markdown("### System Diagnostics")
    
    diagnostics = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    diagnostics.append(("Python Version", python_version, "✅" if sys.version_info >= (3, 7) else "❌"))
    
    # Check Streamlit version
    try:
        import streamlit as st_check
        st_version = st_check.__version__
        diagnostics.append(("Streamlit Version", st_version, "✅"))
    except:
        diagnostics.append(("Streamlit Version", "Unknown", "⚠️"))
    
    # Check critical files
    critical_files = ['core_functions.py', 'admin_interface.py', 'user_interface.py']
    for file in critical_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        diagnostics.append(("File: " + file, "Present" if exists else "Missing", status))
    
    # Check optional files
    optional_files = ['best.pt']
    for file in optional_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "ℹ️"
        diagnostics.append(("Optional: " + file, "Present" if exists else "Not Found", status))
    
    # Display diagnostics table
    for item, value, status in diagnostics:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{item}**")
        with col2:
            st.write(value)
        with col3:
            st.write(status)

if __name__ == "__main__":
    try:
        # Add diagnostic mode
        if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
            st.title("FixMyStreet Diagnostics")
            run_diagnostics()
        else:
            main()
            
    except KeyboardInterrupt:
        st.write("Application stopped by user")
    except Exception as critical_error:
        st.error("Critical Application Error")
        st.exception(critical_error)
        st.markdown("""
        **Recovery Options:**
        1. Refresh the browser page
        2. Restart the Streamlit server
        3. Check the terminal for detailed error messages
        4. Run diagnostics: `streamlit run app.py --diagnostics`
        """)