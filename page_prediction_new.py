"""
Prediction page for making predictions.
Handles loading of model with preprocessor and features.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src_auth import get_current_user_id, require_authentication
from database import get_user_models, get_user_datasets, save_prediction
from utils import load_csv
from preprocessing import DataPreprocessor
import json
import joblib


def load_model_with_preprocessor(model_path):
    """Load model with preprocessor and feature info."""
    try:
        package = joblib.load(model_path)
        if isinstance(package, dict):
            return (
                package.get("model"),
                package.get("preprocessor"),
                package.get("feature_columns"),
                package.get("selected_features")
            )
        st.error("❌ Invalid model format")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None


def show_prediction_page():
    """Display prediction page."""
    st.title("🎯 Make Predictions")
    st.markdown("Use trained models to predict customer churn")
    
    require_authentication()
    user_id = get_current_user_id()
    
    models = get_user_models(user_id)
    datasets = get_user_datasets(user_id)
    
    if not models:
        st.warning("❌ No trained models found. Please train a model first.")
        return
    
    if not datasets:
        st.warning("❌ No datasets found. Please upload a dataset first.")
        return
    
    model_names = {f"{m['model_name']} (Accuracy: {m.get('accuracy', 0):.2%})": m
                   for m in models}
    
    selected_model_label = st.selectbox("Select Model", list(model_names.keys()))
    selected_model = model_names[selected_model_label]
    
    model, preprocessor, feature_columns, selected_features = load_model_with_preprocessor(
        selected_model['model_path']
    )
    
    if model is None:
        st.error("❌ Failed to load model")
        return
    
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
    
    prediction_type = st.radio(
        "Prediction Type",
        ["Single Prediction", "Batch Prediction"],
        horizontal=True
    )
    
    if prediction_type == "Single Prediction":
        show_single_prediction(
            user_id, model, preprocessor, feature_columns,
            selected_features, df_sample, selected_model
        )
    else:
        show_batch_prediction(
            user_id, model, preprocessor, feature_columns, df_sample, selected_model
        )


def show_single_prediction(user_id, model, preprocessor, feature_columns,
                           selected_features, df_sample, selected_model):
    """Show single prediction interface."""
    st.subheader("📝 Enter Customer Information")
    
    if not selected_features:
        st.error("❌ No features available for prediction")
        return
    
    features = [f for f in selected_features if f in df_sample.columns]
    
    if not features:
        st.error("❌ Selected features not found in dataset")
        return
    
    input_data = {}
    cols_per_row = 2
    col_indices = st.columns(cols_per_row)
    
    for i, feature in enumerate(features):
        col_index = i % cols_per_row
        
        with col_indices[col_index]:
            if feature in df_sample.columns:
                if pd.api.types.is_numeric_dtype(df_sample[feature]):
                    min_val = float(df_sample[feature].min())
                    max_val = float(df_sample[feature].max())
                    mean_val = float(df_sample[feature].mean())
                    step = max((max_val - min_val) / 100, 0.01)
                    
                    input_data[feature] = st.slider(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step
                    )
                else:
                    unique_vals = sorted(
                        df_sample[feature].dropna().unique().tolist()
                    )
                    input_data[feature] = st.selectbox(
                        feature,
                        unique_vals
                    )
    
    st.divider()
    
    if st.button("🔮 Make Prediction", use_container_width=True, key="predict_btn"):
        try:
            input_df = pd.DataFrame([input_data])
            
            X_processed = preprocessor.preprocess(input_df, is_training=False)
            X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)
            
            prediction = model.predict(X_processed)[0]
            confidence = model.predict_proba(X_processed)[0].max()
            
            save_prediction(
                user_id, selected_model['model_id'],
                json.dumps(input_data),
                int(prediction), float(confidence)
            )
            
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
            import traceback
            st.error(traceback.format_exc())


def show_batch_prediction(user_id, model, preprocessor, feature_columns, df_sample, selected_model):
    """Show batch prediction interface."""
    st.subheader("📤 Upload CSV for Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with customer data",
        type=['csv'],
        key="batch_upload"
    )
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        
        st.write("**Preview:**")
        st.dataframe(df_input.head(), use_container_width=True)
        
        if st.button("🔮 Make Batch Predictions", use_container_width=True, key="batch_predict_btn"):
            try:
                X_processed = preprocessor.preprocess(df_input, is_training=False)
                X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)
                
                predictions = model.predict(X_processed)
                confidences = model.predict_proba(X_processed).max(axis=1)
                
                results = pd.DataFrame({
                    'prediction': predictions,
                    'confidence': confidences,
                    'risk_level': [get_risk_level(c) for c in confidences]
                })
                
                output_df = pd.concat([df_input.reset_index(drop=True), 
                                      results.reset_index(drop=True)], axis=1)
                
                st.success("✅ Predictions completed!")
                
                st.write("**Prediction Results:**")
                st.dataframe(output_df, use_container_width=True)
                
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
                
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


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
