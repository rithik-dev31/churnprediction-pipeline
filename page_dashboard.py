"""
Dashboard page showing overview and key metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src_auth import get_current_user_id
from database import get_user_datasets, get_user_models, get_user_predictions
from datetime import datetime


def show_dashboard():
    """Display main dashboard."""
    st.title("📊 Dashboard")
    st.markdown("Overview of your ML pipeline and recent activities")
    
    user_id = get_current_user_id()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    datasets = get_user_datasets(user_id)
    models = get_user_models(user_id)
    predictions = get_user_predictions(user_id, limit=100)
    
    with col1:
        st.metric(
            "📁 Datasets Uploaded",
            len(datasets),
            help="Total number of datasets uploaded"
        )
    
    with col2:
        st.metric(
            "🤖 Models Trained",
            len(models),
            help="Total number of models trained"
        )
    
    with col3:
        st.metric(
            "🎯 Predictions Made",
            len(predictions),
            help="Total number of predictions"
        )
    
    with col4:
        avg_accuracy = 0
        if models:
            avg_accuracy = sum(m.get('accuracy', 0) for m in models) / len(models)
        st.metric(
            "📈 Avg Model Accuracy",
            f"{avg_accuracy:.2%}",
            help="Average accuracy of trained models"
        )
    
    st.divider()
    
    # Recent datasets
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Recent Datasets")
        if datasets:
            for dataset in datasets[:5]:
                with st.expander(f"{dataset['dataset_name']} ({dataset['rows']} rows)"):
                    st.write(f"**File Size:** {dataset['file_size'] / 1024:.2f} KB")
                    st.write(f"**Columns:** {dataset['columns']}")
                    st.write(f"**Uploaded:** {dataset['created_at']}")
        else:
            st.info("No datasets uploaded yet. Go to 'Upload Data' to get started!")
    
    with col2:
        st.subheader("🤖 Best Performing Models")
        if models:
            # Sort by accuracy
            models_sorted = sorted(models, key=lambda x: x.get('accuracy', 0), reverse=True)
            for model in models_sorted[:5]:
                with st.expander(f"{model['model_name']} ({model['model_type']})"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
                    with col_b:
                        st.metric("Precision", f"{model.get('precision', 0):.2%}")
                    with col_c:
                        st.metric("F1-Score", f"{model.get('f1_score', 0):.2%}")
        else:
            st.info("No models trained yet. Go to 'Train Models' to start!")
    
    st.divider()
    
    # Predictions trend
    if predictions:
        st.subheader("📈 Recent Predictions")
        
        # Count predictions by date
        predictions_df = pd.DataFrame(predictions)
        predictions_df['date'] = pd.to_datetime(predictions_df['created_at']).dt.date
        predictions_count = predictions_df.groupby('date').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_count['date'],
            y=predictions_count['count'],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Predictions Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Predictions',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent predictions
        st.subheader("Latest Predictions")
        recent_preds = predictions[:10]
        for pred in recent_preds:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Prediction:** {'🔴 Churn' if pred['prediction'] == 1 else '🟢 Retain'}")
            with col2:
                st.write(f"**Confidence:** {pred['confidence']:.2%}")
            with col3:
                st.write(f"**Time:** {pred['created_at'][:10]}")
            st.divider()
    else:
        st.info("No predictions made yet. Go to 'Make Predictions' to start!")
    
    # Statistics cards
    st.subheader("📊 Quick Statistics")
    
    if predictions:
        pred_df = pd.DataFrame(predictions)
        churn_count = (pred_df['prediction'] == 1).sum()
        retain_count = (pred_df['prediction'] == 0).sum()
        avg_confidence = pred_df['confidence'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Churn', 'Retain'],
                values=[churn_count, retain_count],
                hole=0.3
            )])
            fig.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Churn Predictions", churn_count)
            st.metric("Retain Predictions", retain_count)
        
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            st.metric("Total Predictions", len(predictions))
