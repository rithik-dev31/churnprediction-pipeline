"""
Model history and performance tracking page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src_auth import get_current_user_id, require_authentication
from database import get_user_models, get_user_predictions
from utils import load_model_file


def show_history_page():
    """Display model history page."""
    st.title("📋 Model History")
    st.markdown("View and manage your trained models")
    
    require_authentication()
    user_id = get_current_user_id()
    
    # Get models
    models = get_user_models(user_id)
    
    if not models:
        st.info("No models trained yet. Go to 'Train Models' to create one!")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["All Models", "Performance Comparison", "Prediction History"])
    
    # Tab 1: All Models
    with tab1:
        st.subheader("Trained Models")
        
        for model in models:
            with st.expander(f"📊 {model['model_name']} ({model['model_type']})"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
                with col2:
                    st.metric("Precision", f"{model.get('precision', 0):.2%}")
                with col3:
                    st.metric("Recall", f"{model.get('recall', 0):.2%}")
                with col4:
                    st.metric("F1-Score", f"{model.get('f1_score', 0):.2%}")
                
                st.write(f"**Created:** {model['created_at']}")
                st.write(f"**Model Path:** {model['model_path']}")
                
                # Download model
                if st.button(f"📥 Download {model['model_name']}", 
                           key=f"download_{model['model_id']}"):
                    with open(model['model_path'], 'rb') as f:
                        st.download_button(
                            label=f"Download {model['model_name']}",
                            data=f.read(),
                            file_name=f"{model['model_name']}.joblib"
                        )
    
    # Tab 2: Performance Comparison
    with tab2:
        st.subheader("Model Performance Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model in models:
            comparison_data.append({
                'Model': model['model_name'],
                'Accuracy': model.get('accuracy', 0),
                'Precision': model.get('precision', 0),
                'Recall': model.get('recall', 0),
                'F1-Score': model.get('f1_score', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display table
        st.dataframe(df_comparison, use_container_width=True)
        
        # Chart
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_comparison['Model'],
                y=df_comparison[metric]
            ))
        
        fig.update_layout(
            title="Model Metrics Comparison",
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = df_comparison.loc[df_comparison['F1-Score'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model['Model'])
        with col2:
            st.metric("Best F1-Score", f"{best_model['F1-Score']:.2%}")
        with col3:
            st.metric("Best Accuracy", f"{best_model['Accuracy']:.2%}")
    
    # Tab 3: Prediction History
    with tab3:
        st.subheader("Prediction History")
        
        predictions = get_user_predictions(user_id, limit=100)
        
        if predictions:
            pred_df = pd.DataFrame(predictions)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(pred_df))
            with col2:
                churn_count = (pred_df['prediction'] == 1).sum()
                st.metric("Churn Predictions", churn_count)
            with col3:
                retain_count = (pred_df['prediction'] == 0).sum()
                st.metric("Retain Predictions", retain_count)
            with col4:
                avg_confidence = pred_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            # Show predictions
            st.write("**Recent Predictions:**")
            
            # Format for display
            display_df = pred_df[['prediction', 'confidence', 'created_at']].copy()
            display_df['prediction'] = display_df['prediction'].apply(
                lambda x: '🔴 Churn' if x == 1 else '🟢 Retain'
            )
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
            display_df = display_df.rename(columns={
                'prediction': 'Prediction',
                'confidence': 'Confidence',
                'created_at': 'Time'
            })
            
            st.dataframe(display_df, use_container_width=True)
            
            # Trend chart
            pred_df['date'] = pd.to_datetime(pred_df['created_at']).dt.date
            pred_trend = pred_df.groupby('date').size()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_trend.index,
                y=pred_trend.values,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title='Prediction Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Predictions',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions made yet")
