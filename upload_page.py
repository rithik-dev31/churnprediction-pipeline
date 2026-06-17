"""
Upload data page for dataset management.
"""

import streamlit as st
import pandas as pd
from src_auth import get_current_user_id, require_authentication
from database import save_dataset_info
from utils import save_uploaded_file, load_csv, validate_csv, get_dataset_summary


def show_upload_page():
    """Display upload data page."""
    st.title("📁 Upload Dataset")
    st.markdown("Upload and manage your datasets for training and analysis")
    
    require_authentication()
    user_id = get_current_user_id()
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a customer churn dataset"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("📤 Upload", use_container_width=True):
            if uploaded_file is None:
                st.error("Please select a file first")
            else:
                # Validate file
                
                    # Save file
                    filepath = save_uploaded_file(uploaded_file, user_id)
                    
                    if filepath:
                        # Validate CSV
                        is_valid, message = validate_csv(filepath)
                        
                        if is_valid:
                            # Load and analyze
                            df = load_csv(filepath)
                            summary = get_dataset_summary(df)
                            
                            # Save to database
                            success, db_message = save_dataset_info(
                                user_id,
                                uploaded_file.name,
                                filepath,
                                uploaded_file.size,
                                summary['shape'][0],
                                summary['shape'][1]
                            )
                            
                            if success:
                                st.success(f"✅ Dataset uploaded successfully!\n{db_message}")
                                st.rerun()
                            else:
                                st.error(f"Database error: {db_message}")
                        else:
                            st.error(f"Validation failed: {message}")
                    else:
                        st.error("Failed to save file")
            
    
    st.divider()
    
    # Dataset preview
    if uploaded_file:
        st.subheader("📊 Dataset Preview")
        
        df = pd.read_csv(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        st.write("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.write("**First few rows:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.divider()
    
    # Requirements info
    st.subheader("📋 Dataset Requirements")
    st.info("""
    ✓ CSV format required
    ✓ Minimum 10 rows
    ✓ At least 2 columns
    ✓ No completely empty columns
    ✓ Include target variable 'Churn' (0 or 1)
    """)
    
    # Sample dataset info
    st.subheader("📝 Sample Dataset Format")
    st.write("""
    Your dataset should look like:
    
    | CustomerID | Age | Tenure | MonthlyCharges | Churn |
    |---|---|---|---|---|
    | 1 | 32 | 24 | 65.5 | 0 |
    | 2 | 45 | 12 | 89.2 | 1 |
    | 3 | 28 | 36 | 45.0 | 0 |
    """)
