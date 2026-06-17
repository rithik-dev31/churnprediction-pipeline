"""
Prediction module for making predictions on new data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Engine for making predictions with trained models.
    """
    
    def __init__(self, model, preprocessor):
        """
        Initialize prediction engine.
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.prediction_history = []
    
    def predict_single(self, input_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Make a prediction for a single sample.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([input_data])
            
            # Preprocess
            X_processed, _ = self.preprocessor.preprocess(
                df_input, 
                is_training=False
            )
            
            # Predict
            prediction = self.model.predict(X_processed)[0]
            
            # Get probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_processed)[0]
                confidence = max(proba)
            else:
                # For models without predict_proba
                confidence = 0.5 if prediction == 1 else 0.5
            
            # Store history
            self.prediction_history.append({
                'input': input_data,
                'prediction': int(prediction),
                'confidence': float(confidence)
            })
            
            logger.info(f"Single prediction made. Class: {prediction}, Confidence: {confidence:.4f}")
            return int(prediction), float(confidence)
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple samples.
        
        Args:
            df_input: DataFrame with input samples
            
        Returns:
            DataFrame with predictions and confidences
        """
        try:
            # Preprocess
            X_processed, _ = self.preprocessor.preprocess(
                df_input,
                is_training=False
            )
            
            # Predict
            predictions = self.model.predict(X_processed)
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X_processed)
                confidences = np.max(probas, axis=1)
            else:
                confidences = np.ones(len(predictions)) * 0.5
            
            # Create results DataFrame
            results = pd.DataFrame({
                'prediction': predictions.astype(int),
                'confidence': confidences.astype(float),
                'churn_probability': [max(p) if hasattr(self.model, 'predict_proba') else 0.5 for p in (probas if hasattr(self.model, 'predict_proba') else [[0, 0.5]]*len(predictions))]
            })
            
            logger.info(f"Batch predictions made for {len(df_input)} samples")
            return results
        
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def predict_with_interpretation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with interpretation.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and interpretation
        """
        prediction, confidence = self.predict_single(input_data)
        
        interpretation = {
            'prediction': prediction,
            'prediction_label': 'Will Churn' if prediction == 1 else 'Will Not Churn',
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'risk_level': self._get_risk_level(confidence),
            'input_summary': self._summarize_input(input_data)
        }
        
        return interpretation
    
    @staticmethod
    def _get_risk_level(confidence: float) -> str:
        """Determine risk level based on confidence."""
        if confidence >= 0.9:
            return 'Very High'
        elif confidence >= 0.8:
            return 'High'
        elif confidence >= 0.7:
            return 'Medium'
        elif confidence >= 0.6:
            return 'Low'
        else:
            return 'Very Low'
    
    @staticmethod
    def _summarize_input(input_data: Dict[str, Any]) -> Dict[str, str]:
        """Summarize input features."""
        summary = {}
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                summary[key] = f"{value:.2f}"
            else:
                summary[key] = str(value)
        return summary
    
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get prediction history."""
        return self.prediction_history[-limit:]
    
    def clear_history(self):
        """Clear prediction history."""
        self.prediction_history = []


class BatchPredictionProcessor:
    """
    Process batch predictions from CSV files.
    """
    
    def __init__(self, prediction_engine: PredictionEngine):
        """
        Initialize processor.
        
        Args:
            prediction_engine: PredictionEngine instance
        """
        self.engine = prediction_engine
    
    def process_csv(self, filepath: str) -> Tuple[bool, pd.DataFrame, str]:
        """
        Process predictions from CSV file.
        
        Args:
            filepath: Path to input CSV
            
        Returns:
            Tuple of (success, results_df, message)
        """
        try:
            df = pd.read_csv(filepath)
            
            # Remove target column if present
            if 'Churn' in df.columns:
                df = df.drop(columns=['Churn'])
            
            results = self.engine.predict_batch(df)
            
            # Combine input features with predictions
            output_df = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
            
            logger.info(f"Batch processing completed: {len(output_df)} predictions")
            return True, output_df, f"Successfully processed {len(output_df)} samples"
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return False, None, f"Error: {str(e)}"
    
    def save_results(self, results_df: pd.DataFrame, output_path: str) -> bool:
        """
        Save batch prediction results.
        
        Args:
            results_df: Results DataFrame
            output_path: Where to save
            
        Returns:
            Success status
        """
        try:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False


def create_prediction_form_inputs(df_sample: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create form input specification from sample DataFrame.
    
    Args:
        df_sample: Sample DataFrame with feature names and types
        
    Returns:
        Dictionary of form inputs
    """
    form_inputs = {}
    
    for col in df_sample.columns:
        col_type = df_sample[col].dtype
        col_values = df_sample[col].unique()
        
        if col_type in ['int64', 'float64']:
            form_inputs[col] = {
                'type': 'number',
                'min': float(df_sample[col].min()),
                'max': float(df_sample[col].max()),
                'value': float(df_sample[col].mean())
            }
        else:
            form_inputs[col] = {
                'type': 'select',
                'options': sorted([str(v) for v in col_values if pd.notna(v)])
            }
    
    return form_inputs
