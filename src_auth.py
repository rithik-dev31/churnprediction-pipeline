"""
Authentication module for user login/signup and session management.
Uses bcrypt for password hashing and streamlit session state for session management.
"""

import streamlit as st
from database import authenticate_user, create_user, get_user
from typing import Optional, Tuple


def init_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None


def login(username: str, password: str) -> Tuple[bool, str]:
    """
    Authenticate user and set session state.
    
    Args:
        username: Username to login
        password: Password to verify
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    success, user_id, message = authenticate_user(username, password)
    
    if success:
        st.session_state.authenticated = True
        st.session_state.user_id = user_id
        st.session_state.username = username
        st.session_state.user_info = get_user(user_id)
        return True, "Login successful!"
    else:
        return False, message


def signup(username: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Create new user account.
    
    Args:
        username: Username for new account
        email: Email address
        password: Password
        confirm_password: Password confirmation
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Validation
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if not email or "@" not in email:
        return False, "Invalid email address"
    
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    if password != confirm_password:
        return False, "Passwords do not match"
    
    # Create user
    success, message = create_user(username, email, password)
    return success, message


def logout():
    """Logout current user and clear session state."""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.user_info = None


def is_authenticated() -> bool:
    """Check if user is currently authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user_id() -> Optional[int]:
    """Get current authenticated user's ID."""
    return st.session_state.get('user_id', None)


def get_current_username() -> Optional[str]:
    """Get current authenticated user's username."""
    return st.session_state.get('username', None)


def require_authentication():
    """Decorator-like function to check authentication. Raises exception if not authenticated."""
    if not is_authenticated():
        st.error("❌ Please login to access this page")
        st.stop()
