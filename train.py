"""
Model training module.
Trains multiple ML models for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict, Tuple, Any
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and manage multiple ML models.
    """
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.trained_models = {}
        self.training_history = []
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models.
        
        Returns:
            Dictionary of model name to model instance
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                verbosity=0,
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        }
        
        logger.info("Models initialized")
        return self.models
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series) -> Tuple[bool, str]:
        """
        Train a single model.
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if model_name not in self.models:
                return False, f"Model '{model_name}' not found"
            
            model = self.models[model_name]
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            
            logger.info(f"Model trained: {model_name}")
            return True, f"{model_name} trained successfully"
        
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return False, f"Error training {model_name}: {str(e)}"
    
    def train_all_models(self, X_train: pd.DataFrame, 
                        y_train: pd.Series) -> Dict[str, Tuple[bool, str]]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        for model_name in self.models.keys():
            success, message = self.train_model(model_name, X_train, y_train)
            results[model_name] = (success, message)
        
        logger.info(f"All models trained. Successful: {sum(1 for s, _ in results.values() if s)}/{len(results)}")
        return results
    
    def get_model(self, model_name: str) -> Any:
        """Get trained model."""
        return self.trained_models.get(model_name, None)
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with a trained model.
        
        Args:
            model_name: Name of trained model
            X: Features to predict
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        return model.predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of trained model
            X: Features to predict
            
        Returns:
            Probability predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # For SVM with probability=True
            return model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    def cross_validate(self, model_name: str, X: pd.DataFrame, 
                      y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of model
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'fold_scores': scores.tolist()
        }
    
    def kfold_validation(self, model_name: str, X: pd.DataFrame, 
                        y: pd.Series, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform K-Fold cross-validation.
        
        Args:
            model_name: Name of model
            X: Features
            y: Target
            n_splits: Number of splits
            
        Returns:
            Dictionary with fold scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        
        return {
            'n_splits': n_splits,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'fold_scores': scores.tolist()
        }
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """
        Save trained model to file.
        
        Args:
            model_name: Name of model to save
            filepath: Where to save
            
        Returns:
            Success status
        """
        try:
            if model_name not in self.trained_models:
                logger.error(f"Model '{model_name}' not trained")
                return False
            
            model = self.trained_models[model_name]
            joblib.dump(model, filepath)
            logger.info(f"Model saved: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            model_name: Name to assign to model
            filepath: Where to load from
            
        Returns:
            Success status
        """
        try:
            model = joblib.load(filepath)
            self.trained_models[model_name] = model
            logger.info(f"Model loaded: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


class ModelHyperparameterTuner:
    """
    Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
    """
    
    def __init__(self):
        """Initialize tuner."""
        self.best_params = {}
        self.best_models = {}
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get parameter grids for grid search.
        
        Returns:
            Dictionary of parameter grids for each model
        """
        return {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [100, 500, 1000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    
    def grid_search(self, model_name: str, model, param_grid: Dict, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   cv: int = 5) -> Tuple[Any, Dict]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model_name: Name of model
            model: Model instance
            param_grid: Parameter grid
            X_train: Training features
            y_train: Training target
            cv: Number of folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        from sklearn.model_selection import GridSearchCV
        
        try:
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params[model_name] = grid_search.best_params_
            self.best_models[model_name] = grid_search.best_estimator_
            
            logger.info(f"Grid search completed for {model_name}")
            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best score: {grid_search.best_score_}")
            
            return grid_search.best_estimator_, grid_search.best_params_
        
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise
    
    def random_search(self, model_name: str, model, param_dist: Dict,
                     X_train: pd.DataFrame, y_train: pd.Series,
                     n_iter: int = 20, cv: int = 5) -> Tuple[Any, Dict]:
        """
        Perform randomized search for hyperparameter tuning.
        
        Args:
            model_name: Name of model
            model: Model instance
            param_dist: Parameter distribution
            X_train: Training features
            y_train: Training target
            n_iter: Number of iterations
            cv: Number of folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        try:
            random_search = RandomizedSearchCV(
                model, param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            random_search.fit(X_train, y_train)
            
            self.best_params[model_name] = random_search.best_params_
            self.best_models[model_name] = random_search.best_estimator_
            
            logger.info(f"Random search completed for {model_name}")
            logger.info(f"Best params: {random_search.best_params_}")
            logger.info(f"Best score: {random_search.best_score_}")
            
            return random_search.best_estimator_, random_search.best_params_
        
        except Exception as e:
            logger.error(f"Error in random search: {str(e)}")
            raise
    
    def get_best_model(self, model_name: str) -> Any:
        """Get best tuned model."""
        return self.best_models.get(model_name, None)
    
    def get_best_params(self, model_name: str) -> Dict:
        """Get best parameters."""
        return self.best_params.get(model_name, {})
