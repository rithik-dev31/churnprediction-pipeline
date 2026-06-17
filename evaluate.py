"""
Model evaluation module.
Evaluates trained models with various metrics and visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.confusion_matrices = {}
        self.roc_curves = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"Metrics calculated for {model_name}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model") -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm
        
        logger.info(f"Confusion matrix calculated for {model_name}")
        return cm
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model") -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, zero_division=0)
        logger.info(f"Classification report for {model_name}:\n{report}")
        return report
    
    def get_roc_curve_data(self, y_true: np.ndarray, y_proba: np.ndarray,
                          model_name: str = "Model") -> Dict[str, Any]:
        """
        Calculate ROC curve data.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of model
            
        Returns:
            Dictionary with ROC curve data
        """
        try:
            # Handle binary classification
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            self.roc_curves[model_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
            
            logger.info(f"ROC curve calculated for {model_name}, AUC: {roc_auc:.4f}")
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'thresholds': thresholds
            }
        except Exception as e:
            logger.error(f"Error calculating ROC curve: {str(e)}")
            return {}
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "Model") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred, model_name)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of model
            
        Returns:
            Matplotlib figure
        """
        roc_data = self.get_roc_curve_data(y_true, y_proba, model_name)
        
        if not roc_data:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(roc_data['fpr'], roc_data['tpr'], 
               label=f"AUC = {roc_data['auc']:.4f}", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_metrics_comparison(self, models_metrics: Dict[str, Dict]) -> plt.Figure:
        """
        Plot comparison of metrics across models.
        
        Args:
            models_metrics: Dictionary of model names to metrics
            
        Returns:
            Matplotlib figure
        """
        df = pd.DataFrame(models_metrics).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(loc='best')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_comparison_plotly(self, models_metrics: Dict[str, Dict]) -> go.Figure:
        """
        Plot comparison of metrics across models using Plotly.
        
        Args:
            models_metrics: Dictionary of model names to metrics
            
        Returns:
            Plotly figure
        """
        data = []
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_list:
            values = [models_metrics.get(model, {}).get(metric, 0) 
                     for model in models_metrics.keys()]
            data.append(go.Bar(name=metric.replace('_', ' ').title(), 
                              x=list(models_metrics.keys()), 
                              y=values))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            xaxis_title='Model',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_proba: Optional[np.ndarray] = None,
                                  model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of model
            
        Returns:
            Dictionary with complete evaluation report
        """
        report = {
            'model_name': model_name,
            'metrics': self.calculate_metrics(y_true, y_pred, model_name),
            'confusion_matrix': self.get_confusion_matrix(y_true, y_pred, model_name).tolist(),
            'classification_report': self.get_classification_report(y_true, y_pred, model_name)
        }
        
        if y_proba is not None:
            report['roc_curve'] = self.get_roc_curve_data(y_true, y_proba, model_name)
            # Convert numpy arrays to lists for JSON serialization
            if 'roc_curve' in report and report['roc_curve']:
                report['roc_curve']['fpr'] = report['roc_curve']['fpr'].tolist()
                report['roc_curve']['tpr'] = report['roc_curve']['tpr'].tolist()
        
        logger.info(f"Evaluation report generated for {model_name}")
        return report


def compare_models(evaluations: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model evaluations.
    
    Args:
        evaluations: Dictionary of model evaluations
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = {}
    
    for model_name, eval_data in evaluations.items():
        if 'metrics' in eval_data:
            comparison_data[model_name] = eval_data['metrics']
    
    df_comparison = pd.DataFrame(comparison_data).T
    df_comparison = df_comparison.round(4)
    
    logger.info("Model comparison completed")
    return df_comparison


def get_best_model(evaluations: Dict[str, Dict], metric: str = 'f1_score') -> Tuple[str, float]:
    """
    Get best model based on specified metric.
    
    Args:
        evaluations: Dictionary of model evaluations
        metric: Metric to use for comparison
        
    Returns:
        Tuple of (best_model_name, best_score)
    """
    best_model = None
    best_score = -1
    
    for model_name, eval_data in evaluations.items():
        if 'metrics' in eval_data and metric in eval_data['metrics']:
            score = eval_data['metrics'][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
    
    logger.info(f"Best model: {best_model} with {metric}: {best_score:.4f}")
    return best_model, best_score
