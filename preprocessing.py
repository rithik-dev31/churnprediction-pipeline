"""
Data preprocessing module for ML pipeline.
Handles missing values, encoding, scaling, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing class.
    Handles all preprocessing steps in a sklearn-compatible pipeline.
    """

    def __init__(self, target_col: str = 'Exited'):
        """
        Initialize preprocessor.

        Args:
            target_col: Name of target column (default: 'Exited')
        """
        self.target_col = target_col
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.preprocessor = None
        self.numerical_features = None
        self.categorical_features = None

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features — must exactly match feature_engineering.py.
        This is called when create_features=True (e.g. prediction path).
        """
        df_copy = df.copy()

        # Age_Group — matches feature_engineering._add_age_group
        if 'Age' in df_copy.columns:
            df_copy['Age_Group'] = pd.cut(
                df_copy['Age'],
                bins=[0, 30, 40, 50, 100],
                labels=['Young', 'Mid Adult', 'Senior', 'Elderly'],
                include_lowest=True,
            )

        # credit_label — matches feature_engineering._add_credit_label
        if 'CreditScore' in df_copy.columns:
            df_copy['credit_label'] = pd.cut(
                df_copy['CreditScore'],
                bins=[0, 600, 700, 800, 900],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True,
            )

        # Credit_Group — matches feature_engineering._add_credit_group
        if 'CreditScore' in df_copy.columns:
            df_copy['Credit_Group'] = pd.cut(
                df_copy['CreditScore'],
                bins=[0, 300, 580, 670, 740, 800, 850],
                labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional', 'Elite'],
                include_lowest=True,
            )

        # balance_label — matches feature_engineering._add_balance_label
        if 'Balance' in df_copy.columns:
            df_copy['balance_label'] = pd.cut(
                df_copy['Balance'],
                bins=[-1, 0, 50_000, 100_000, 150_000, 300_000],
                labels=['Zero', 'Low', 'Medium', 'High', 'Very High'],
                include_lowest=True,
            )

        # salary_label — matches feature_engineering._add_salary_label
        if 'EstimatedSalary' in df_copy.columns:
            df_copy['salary_label'] = pd.cut(
                df_copy['EstimatedSalary'],
                bins=[0, 50_000, 100_000, 150_000, 200_000],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True,
            )

        return df_copy

    def _get_categorical_cols(self, df: pd.DataFrame) -> pd.Index:
        """
        Get categorical columns compatible with pandas 3.x.
        Covers object/str dtypes (kind='O') and category dtype (from pd.cut/pd.Categorical).
        Avoids passing 'str' to select_dtypes which raises an error in pandas 3.x.

        Args:
            df: Input DataFrame

        Returns:
            Index of categorical column names
        """
        # dtype.kind == 'O' captures both legacy 'object' and new 'str' dtypes in pandas 3
        obj_cols = [col for col in df.columns if df[col].dtype.kind == 'O']
        # select_dtypes for 'category' (produced by pd.cut / pd.Categorical)
        cat_cols = df.select_dtypes(include=['category']).columns.tolist()
        # Merge, deduplicate, preserve column order
        seen = set()
        combined = []
        for col in obj_cols + cat_cols:
            if col not in seen:
                seen.add(col)
                combined.append(col)
        return pd.Index(combined)

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            df: Input DataFrame
            method: 'mean', 'median', 'drop', or 'forward_fill'

        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()

        if method == 'drop':
            df_copy = df_copy.dropna()

        elif method == 'forward_fill':
            # Use ffill()/bfill() — fillna(method=...) is deprecated in pandas 3.x
            df_copy = df_copy.ffill().bfill()

        else:
            numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
            categorical_cols = self._get_categorical_cols(df_copy)

            if method == 'mean':
                df_copy[numerical_cols] = df_copy[numerical_cols].fillna(
                    df_copy[numerical_cols].mean()
                )
                # FIX: mode() returns a DataFrame; use .iloc[0] to get first row (actual mode values)
                if len(categorical_cols) > 0:
                    df_copy[categorical_cols] = df_copy[categorical_cols].fillna(
                        df_copy[categorical_cols].mode().iloc[0]
                    )

            elif method == 'median':
                df_copy[numerical_cols] = df_copy[numerical_cols].fillna(
                    df_copy[numerical_cols].median()
                )
                # FIX: same as above
                if len(categorical_cols) > 0:
                    df_copy[categorical_cols] = df_copy[categorical_cols].fillna(
                        df_copy[categorical_cols].mode().iloc[0]
                    )

        logger.info(f"Missing values handled using '{method}' method")
        return df_copy

    def encode_categorical(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            method: 'label' or 'onehot'

        Returns:
            DataFrame with encoded features
        """
        df_copy = df.copy()
        # FIX: include 'category' dtype (produced by pd.cut) alongside object/str
        categorical_cols = self._get_categorical_cols(df_copy)

        if method == 'label':
            for col in categorical_cols:
                # Cast to str to handle category dtype and any NaN edge cases
                col_values = df_copy[col].astype(str)
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(col_values)
                    self.label_encoders[col] = le
                else:
                    df_copy[col] = self.label_encoders[col].transform(col_values)

        elif method == 'onehot':
            # Cast category columns to str before get_dummies to avoid dtype issues
            for col in categorical_cols:
                df_copy[col] = df_copy[col].astype(str)
            df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)

        logger.info(f"Categorical variables encoded using '{method}' method")
        return df_copy

    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            method: 'standard' or 'minmax'

        Returns:
            Scaled DataFrame
        """
        df_copy = df.copy()
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column from scaling
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)

        if not numerical_cols:
            logger.warning("No numerical columns found for scaling")
            return df_copy

        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
            else:
                df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])

        elif method == 'minmax':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                df_copy[numerical_cols] = self.scaler.fit_transform(df_copy[numerical_cols])
            else:
                df_copy[numerical_cols] = self.scaler.transform(df_copy[numerical_cols])

        logger.info(f"Features scaled using '{method}' scaler")
        return df_copy

    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top K features using SelectKBest.

        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select

        Returns:
            Tuple of (reduced_features, selected_feature_names)
        """
        k = min(k, X.shape[1])

        if self.feature_selector is None:
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)

        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()

        logger.info(f"Selected {k} best features: {self.selected_features}")

        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features

    def preprocess(self, df: pd.DataFrame, is_training: bool = True,
                   missing_method: str = 'mean', encoding_method: str = 'label',
                   scaling_method: str = 'standard', feature_selection_k: Optional[int] = None,
                   create_features: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete preprocessing pipeline with optional feature engineering.

        Args:
            df: Input DataFrame
            is_training: If True, fit preprocessors; if False, transform only
            missing_method: Method for handling missing values
            encoding_method: Method for encoding categorical variables
            scaling_method: Method for scaling features
            feature_selection_k: Number of features to select (None to skip)
            create_features: Whether to create engineered features

        Returns:
            Tuple of (processed_features, target) if target exists, else (processed_features, None)
        """
        df_processed = df.copy()
        target = None

        # Separate target if it exists
        if self.target_col in df_processed.columns:
            target = df_processed[self.target_col].copy()
            df_processed = df_processed.drop(columns=[self.target_col])

        # Create engineered features BEFORE handling missing values
        if create_features:
            df_processed = self.create_engineered_features(df_processed)

        # Handle missing values
        df_processed = self.handle_missing_values(df_processed, method=missing_method)

        # Encode categorical variables
        df_processed = self.encode_categorical(df_processed, method=encoding_method)

        # Scale features
        df_processed = self.scale_features(df_processed, method=scaling_method)

        # Feature selection
        if feature_selection_k is not None and is_training and target is not None:
            df_processed, _ = self.feature_selection(df_processed, target, k=feature_selection_k)
        elif self.selected_features is not None:
            df_processed = df_processed[self.selected_features]

        logger.info("Data preprocessing completed")
        return df_processed, target

    def save(self, filepath: str) -> bool:
        """Save preprocessor to file."""
        try:
            joblib.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': self.selected_features
            }, filepath)
            logger.info(f"Preprocessor saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            return False

    def load(self, filepath: str) -> bool:
        """Load preprocessor from file."""
        try:
            data = joblib.load(filepath)
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.feature_selector = data['feature_selector']
            self.selected_features = data['selected_features']
            logger.info(f"Preprocessor loaded: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            return False


def train_test_split_stratified(X: pd.DataFrame, y: pd.Series,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test