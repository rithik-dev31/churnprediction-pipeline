"""
Prediction page for making predictions - FIXED VERSION.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src_auth import get_current_user_id, require_authentication
from database import get_user_models, get_user_datasets, save_prediction
from utils import load_csv, load_model_file
from preprocessing import DataPreprocessor
from predict import PredictionEngine
import json
import joblib
import os
import pickle


def load_model_with_preprocessor(model_path):
    """
    Load model along with its preprocessor.
    Tries multiple loading methods to handle different save formats.
    """
    try:
        # Try loading as pickle first
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except:
        try:
            # Try joblib
            model = joblib.load(model_path)
            return model, None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None


def get_feature_columns(df):
    """Get all feature columns (excluding Churn)."""
    features = [col for col in df.columns if col.lower() != 'churn']
    return features


def show_prediction_page():
    """Display prediction page."""
    st.title("🎯 Make Predictions")
    st.markdown("Use trained models to predict customer churn")
    
    require_authentication()
    user_id = get_current_user_id()
    
    # Get user models and datasets
    models = get_user_models(user_id)
    datasets = get_user_datasets(user_id)
    
    if not models:
        st.warning("❌ No trained models found. Please train a model first.")
        return
    
    if not datasets:
        st.warning("❌ No datasets found. Please upload a dataset first.")
        return
    
    # Select model
    model_names = {f"{m['model_name']} (Accuracy: {m.get('accuracy', 0):.2%})": m
                   for m in models}
    
    selected_model_label = st.selectbox(
        "Select Model",
        list(model_names.keys())
    )
    
    selected_model = model_names[selected_model_label]
    model, _ = load_model_with_preprocessor(selected_model['model_path'])
    
    if model is None:
        st.error("❌ Failed to load model")
        return
    
    # Get dataset for feature information
    dataset_names = {f"{d['dataset_name']}": d['file_path'] for d in datasets}
    selected_dataset_label = st.selectbox(
        "Select Dataset (for feature reference)",
        list(dataset_names.keys())
    )
    
    dataset_path = dataset_names[selected_dataset_label]
    df_sample = load_csv(dataset_path)
    
    if df_sample is None:
        st.error("❌ Failed to load dataset")
        return
    
    st.divider()
    
    # Prediction type selection
    prediction_type = st.radio(
        "Prediction Type",
        ["Single Prediction", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_type == "Single Prediction":
        show_single_prediction(user_id, model, df_sample, selected_model)
    else:
        show_batch_prediction(user_id, model, df_sample, selected_model)


def show_single_prediction(user_id, model, df_sample, selected_model):
    """Show single prediction interface."""
    st.subheader("📝 Enter Customer Information")
    
    # Get feature columns from sample
    features = get_feature_columns(df_sample)
    
    if not features:
        st.error("❌ No features found in dataset")
        return
    
    st.info(f"📌 Expected Features: {', '.join(features)}")
    
    # Create input form
    input_data = {}
    
    cols_per_row = 2
    col_indices = st.columns(cols_per_row)
    
    for i, feature in enumerate(features):
        col_index = i % cols_per_row
        with col_indices[col_index]:
            col_type = df_sample[feature].dtype
            
            if col_type in ['int64', 'float64']:
                try:
                    min_val = float(df_sample[feature].min())
                    max_val = float(df_sample[feature].max())
                    mean_val = float(df_sample[feature].mean())
                    
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
                except Exception as e:
                    st.error(f"Error with {feature}: {str(e)}")
            else:
                try:
                    unique_vals = sorted(df_sample[feature].dropna().unique())
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        unique_vals,
                        key=f"select_{feature}"
                    )
                except Exception as e:
                    st.error(f"Error with {feature}: {str(e)}")
    
    st.divider()
    
    # Predict button
    if st.button("🔮 Make Prediction", use_container_width=True, key="predict_btn"):
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([input_data])
            
            # Preprocess and predict
            preprocessor = DataPreprocessor('Churn')
            
            # Preprocess the input (use same methods as training)
            X_processed, _ = preprocessor.preprocess(
                df_input,
                is_training=False,
                missing_method='mean',
                encoding_method='label',
                scaling_method='standard'
            )
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)[0]
                confidence = max(proba)
            else:
                confidence = 0.5
            
            # Save to database
            save_prediction(
                user_id, selected_model['model_id'],
                json.dumps(input_data),
                int(prediction), float(confidence)
            )
            
            # Display results
            st.divider()
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("🔴 HIGH RISK - Likely to Churn")
                else:
                    st.success("🟢 LOW RISK - Likely to Retain")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col3:
                risk_level = get_risk_level(confidence)
                st.metric("Risk Level", risk_level)
            
            # Recommendation
            st.divider()
            st.subheader("💡 Recommendation")
            
            if prediction == 1:
                st.warning("""
                This customer is predicted to churn. Consider:
                - 📞 Personalized outreach and engagement
                - 💰 Special retention offers or discounts
                - 📊 Review their usage patterns and pain points
                - 👥 Dedicated account management
                """)
            else:
                st.info("""
                This customer is likely to stay. Continue to:
                - 📈 Monitor satisfaction levels
                - 🎁 Offer upsell opportunities
                - 👍 Maintain quality service
                - 📧 Regular engagement communications
                """)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("**Solution:** Make sure your input matches the training data format. Check:")
            st.code(f"Error details: {str(e)}", language="python")


def show_batch_prediction(user_id, model, df_sample, selected_model):
    """Show batch prediction interface."""
    st.subheader("📤 Upload CSV for Batch Prediction")
    
    # Show expected columns
    expected_cols = get_feature_columns(df_sample)
    st.info(f"📌 Expected columns in CSV: {', '.join(expected_cols)}")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with customer data",
        type=['csv'],
        key="batch_upload"
    )
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        
        st.write("**Preview:**")
        st.dataframe(df_input.head(), use_container_width=True)
        
        # Validate columns
        missing_cols = set(expected_cols) - set(df_input.columns)
        extra_cols = set(df_input.columns) - set(expected_cols) - {'Churn'}
        
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            return
        
        if extra_cols:
            st.warning(f"⚠️ Extra columns (will be ignored): {extra_cols}")
        
        if st.button("🔮 Make Batch Predictions", use_container_width=True, key="batch_predict_btn"):
            try:
                # Select only expected columns
                df_input_clean = df_input[expected_cols].copy()
                
                # Preprocess
                preprocessor = DataPreprocessor('Churn')
                X_processed, _ = preprocessor.preprocess(
                    df_input_clean,
                    is_training=False,
                    missing_method='mean',
                    encoding_method='label',
                    scaling_method='standard'
                )
                
                # Make predictions
                predictions = model.predict(X_processed)
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_processed)
                    confidences = np.max(probas, axis=1)
                else:
                    confidences = np.ones(len(predictions)) * 0.5
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'prediction': predictions.astype(int),
                    'prediction_label': ['Churn' if p == 1 else 'Retain' for p in predictions],
                    'confidence': confidences.astype(float)
                })
                
                # Combine with input
                output_df = pd.concat([df_input.reset_index(drop=True), 
                                      results.reset_index(drop=True)], axis=1)
                
                st.success("✅ Predictions completed!")
                
                # Display results
                st.write("**Prediction Results:**")
                st.dataframe(output_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                churn_count = (results['prediction'] == 1).sum()
                retain_count = (results['prediction'] == 0).sum()
                avg_confidence = results['confidence'].mean()
                
                with col1:
                    st.metric("Predicted Churn", churn_count)
                with col2:
                    st.metric("Predicted Retain", retain_count)
                with col3:
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Download results
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
                st.info("**Troubleshooting:**")
                st.code(f"Error details: {str(e)}", language="python")
                st.write("Make sure all required columns are present and have correct data types")


def get_risk_level(confidence: float) -> str:
    """Determine risk level based on confidence."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.7:
        return "Medium"
    elif confidence >= 0.6:
        return "Low"
    else:
        return "Very Low"
