"""
Main Streamlit application entry point.
Customer Churn Prediction System.
"""

import streamlit as st
import os
from datetime import datetime
from src_auth import init_session_state, is_authenticated, get_current_username
from database import init_database
from utils import ensure_directories
import logging

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

init_database()
ensure_directories()
init_session_state()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def show_home():
    """Display home page."""
    st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction System</h1>", 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Welcome! 👋
        
        This is a comprehensive Machine Learning pipeline for predicting customer churn.
        
        ### Features:
        - 🔐 Secure user authentication
        - 📈 Interactive data exploration (EDA)
        - 🤖 Multiple ML models (6 algorithms)
        - ⚙️ Hyperparameter tuning
        - 🎯 Real-time predictions
        - 📊 Model monitoring & analytics
        - 💾 Model persistence & management
        """)
    
    with col2:
        st.markdown("""
        ### Supported Models:
        1. **Logistic Regression** - Fast, interpretable
        2. **Random Forest** - Robust ensemble method
        3. **Decision Tree** - Easy to interpret
        4. **XGBoost** - High performance gradient boosting
        5. **Support Vector Machine** - Powerful classifier
        6. **K-Nearest Neighbors** - Simple & effective
        
        ### Get Started:
        1. Sign up or login from the sidebar
        2. Upload your dataset
        3. Explore the data
        4. Train models
        5. Make predictions
        """)
    
    st.divider()
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Available", 6, "🤖")
    with col2:
        st.metric("Evaluation Metrics", 4, "📊")
    with col3:
        st.metric("Tuning Methods", 2, "⚙️")
    with col4:
        st.metric("Authentication", "✓", "🔐")


def show_auth_pages():
    """Show login/signup pages."""
    with st.sidebar:
        st.markdown("---")
        auth_mode = st.radio(
            "Choose Mode",
            ["Login", "Signup"],
            key="auth_mode"
        )
        
        if auth_mode == "Login":
            show_login_page()
        else:
            show_signup_page()


def show_login_page():
    """Display login page."""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please fill in all fields")
            else:
                from src_auth import login
                success, message = login(username, password)
                
                if success:
                    st.success(message)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)


def show_signup_page():
    """Display signup page."""
    st.title("📝 Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            from src_auth import signup
            success, message = signup(username, email, password, confirm_password)
            
            if success:
                st.success(message)
                st.info("Account created! Please login with your credentials.")
            else:
                st.error(message)


def show_authenticated_ui():
    """Show main UI for authenticated users."""
    username = get_current_username()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {username}")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Upload Data", "Explore Data", "Train Models", "Make Predictions", "Model History", "Settings"],
            key="page_nav"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            from src_auth import logout
            logout()
            st.rerun()
    
    # Route to pages
    if page == "Dashboard":
        from page_dashboard import show_dashboard
        show_dashboard()
    elif page == "Upload Data":
        from upload_page import show_upload_page
        show_upload_page()
    elif page == "Explore Data":
        from page_eda import show_eda_page
        show_eda_page()
    elif page == "Train Models":
        from page_training import show_training_page
        show_training_page()
    elif page == "Make Predictions":
        from page_prediction import show_prediction_page
        show_prediction_page()
    elif page == "Model History":
        from page_history import show_history_page
        show_history_page()
    elif page == "Settings":
        from page_settings import show_settings_page
        show_settings_page()


def main():
    """Main application entry point."""
    if is_authenticated():
        show_authenticated_ui()
    else:
        show_home()
        show_auth_pages()


if __name__ == "__main__":
    main()
