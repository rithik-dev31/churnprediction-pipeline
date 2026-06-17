"""
Utility functions for the ML pipeline.
Includes helpers for data handling, visualization, and general utilities.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== FILE OPERATIONS ====================

def ensure_directories():
    """Ensure all required directories exist."""
    directories = ['models', 'datasets', 'logs', 'database']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories ensured")


def save_uploaded_file(uploaded_file, user_id: int) -> Optional[str]:
    """
    Save uploaded CSV file to datasets directory.
    
    Args:
        uploaded_file: Streamlit uploaded file
        user_id: Current user ID
        
    Returns:
        Path to saved file or None
    """
    try:
        ensure_directories()
        filename = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        filepath = os.path.join('datasets', filename)
        
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None


def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load CSV file into pandas DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame or None if error
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"CSV loaded: {filepath} - Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return None


def save_model_file(model, filepath: str) -> bool:
    """
    Save model using joblib.
    
    Args:
        model: Model object to save
        filepath: Where to save the model
        
    Returns:
        Success status
    """
    try:
        import joblib
        ensure_directories()
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False


def load_model_file(filepath: str):
    """
    Load model using joblib.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Model object or None
    """
    try:
        import joblib
        model = joblib.load(filepath)
        logger.info(f"Model loaded: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


# ==================== DATA ANALYSIS ====================

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary of dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numerical columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def detect_outliers(data: pd.Series, method: str = 'iqr') -> np.ndarray:
    """
    Detect outliers using IQR or Z-score method.
    
    Args:
        data: Series to check for outliers
        method: 'iqr' or 'zscore'
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
    else:  # zscore
        from scipy import stats
        outliers = np.abs(stats.zscore(data.dropna())) > 3
    
    return outliers


def correlation_with_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Calculate correlation of all features with target variable.
    
    Args:
        df: DataFrame with all features and target
        target_col: Name of target column
        
    Returns:
        Series of correlations sorted by absolute value
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    numerical_cols = get_numerical_columns(df)
    correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    return correlations


# ==================== VALIDATION ====================

def validate_csv(filepath: str, min_rows: int = 10) -> tuple[bool, str]:
    """
    Validate CSV file.
    
    Args:
        filepath: Path to CSV file
        min_rows: Minimum required rows
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    try:
        df = pd.read_csv(filepath)
        
        if len(df) < min_rows:
            return False, f"CSV must have at least {min_rows} rows. Found {len(df)}"
        
        if df.shape[1] < 2:
            return False, "CSV must have at least 2 columns"
        
        if df.isnull().all().any():
            return False, "Some columns are entirely empty"
        
        return True, "CSV is valid"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"


# ==================== CACHING ====================

@st.cache_data
def cached_load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Load CSV with Streamlit caching."""
    return load_csv(filepath)


@st.cache_data
def cached_get_summary(filepath: str) -> Optional[Dict]:
    """Get summary with Streamlit caching."""
    df = load_csv(filepath)
    if df is not None:
        return get_dataset_summary(df)
    return None


# ==================== PROGRESS & NOTIFICATIONS ====================

def show_progress(value: float, text: str = ""):
    """Show progress bar."""
    progress_bar = st.progress(0)
    for i in range(int(value * 100)):
        progress_bar.progress(i / 100)
    progress_bar.progress(1.0)
    if text:
        st.success(text)


def log_event(event: str, details: str = ""):
    """Log application event."""
    logger.info(f"{event}: {details}")
    with open('logs/events.log', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - {event} - {details}\n")
