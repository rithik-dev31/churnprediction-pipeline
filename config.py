"""
Application configuration and constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== APPLICATION SETTINGS ====================

APP_NAME = os.getenv("APP_NAME", "Customer Churn Prediction System")
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ==================== DATABASE SETTINGS ====================

DATABASE_PATH = os.getenv("DATABASE_PATH", "database/churn_prediction.db")
DATABASE_TIMEOUT = 30

# ==================== PATHS ====================

MODELS_DIR = os.getenv("MODELS_DIR", "models")
DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# Ensure directories exist
for directory in [MODELS_DIR, DATASETS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== MACHINE LEARNING SETTINGS ====================

# Model training
DEFAULT_TEST_SIZE = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_CV_FOLDS = int(os.getenv("DEFAULT_CV_FOLDS", "5"))
DEFAULT_RANDOM_STATE = int(os.getenv("DEFAULT_RANDOM_STATE", "42"))

# Feature processing
DEFAULT_SCALING_METHOD = os.getenv("DEFAULT_SCALING_METHOD", "standard")
DEFAULT_ENCODING_METHOD = os.getenv("DEFAULT_ENCODING_METHOD", "label")
DEFAULT_MISSING_VALUE_METHOD = os.getenv("DEFAULT_MISSING_VALUE_METHOD", "mean")

# Feature selection
ENABLE_FEATURE_SELECTION = os.getenv("ENABLE_FEATURE_SELECTION", "False").lower() == "true"
DEFAULT_FEATURE_COUNT = int(os.getenv("DEFAULT_FEATURE_COUNT", "10"))

# ==================== MODEL CONFIGURATIONS ====================

# Model hyperparameters
MODEL_CONFIGS = {
    'Logistic Regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'n_jobs': -1
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': DEFAULT_RANDOM_STATE,
        'n_jobs': -1
    },
    'Decision Tree': {
        'max_depth': 10,
        'min_samples_split': 2,
        'criterion': 'gini',
        'random_state': DEFAULT_RANDOM_STATE
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'random_state': DEFAULT_RANDOM_STATE,
        'verbosity': 0
    },
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'random_state': DEFAULT_RANDOM_STATE
    },
    'KNN': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'metric': 'euclidean',
        'n_jobs': -1
    }
}

# ==================== DATA VALIDATION ====================

# File upload
MAX_UPLOAD_SIZE_MB = int(os.getenv("STREAMLIT_MAX_UPLOAD_SIZE", "200"))
ALLOWED_FILE_TYPES = ['.csv']
MIN_ROWS_REQUIRED = 10
MIN_COLUMNS_REQUIRED = 2

# Target column configuration
TARGET_COLUMN = 'Churn'
TARGET_VALUES = {0: 'No Churn', 1: 'Churn'}

# ==================== SECURITY SETTINGS ====================

# Password hashing
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))

# Session management
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour

# ==================== STREAMLIT SETTINGS ====================

STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "light")
STREAMLIT_LOGGER_LEVEL = os.getenv("STREAMLIT_LOGGER_LEVEL", "info")

# ==================== EVALUATION METRICS ====================

# Metrics to calculate
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc'
]

# ==================== COLOR SCHEME ====================

COLORS = {
    'primary': '#1f77b4',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'churn': '#e74c3c',
    'no_churn': '#2ecc71'
}

# ==================== MESSAGES ====================

MESSAGES = {
    'login_success': '✅ Login successful!',
    'login_failed': '❌ Invalid username or password',
    'signup_success': '✅ Account created successfully!',
    'signup_failed': '❌ Failed to create account',
    'logout_success': '✅ Logged out successfully',
    'upload_success': '✅ Dataset uploaded successfully!',
    'upload_failed': '❌ Failed to upload dataset',
    'train_success': '✅ Models trained successfully!',
    'train_failed': '❌ Training failed',
    'prediction_success': '✅ Prediction completed!',
    'prediction_failed': '❌ Prediction failed',
    'model_saved': '✅ Model saved successfully',
    'model_load_failed': '❌ Failed to load model',
    'please_login': '❌ Please login to access this page',
    'select_dataset': '⚠️ Please select a dataset first',
    'no_models': '⚠️ No trained models found'
}

# ==================== FEATURE DESCRIPTIONS ====================

FEATURE_DESCRIPTIONS = {
    'Churn': 'Customer churn indicator (0: No churn, 1: Churn)',
    'Age': 'Customer age in years',
    'Tenure': 'Number of months customer has been with company',
    'MonthlyCharges': 'Monthly charge amount in dollars',
    'TotalCharges': 'Total charges paid by customer'
}

# ==================== API CONFIGURATION ====================

# Rate limiting (if implementing API)
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 3600  # seconds

# ==================== EXPORT CONFIGURATION ====================

# Export formats
EXPORT_FORMATS = ['csv', 'xlsx', 'json']

# PDF Report settings
PDF_MARGIN = 10
PDF_PAGE_HEIGHT = 297
PDF_PAGE_WIDTH = 210

# ==================== LOGGING CONFIGURATION ====================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ==================== CACHE SETTINGS ====================

# Caching duration in seconds
CACHE_DURATION_DATA = 3600  # 1 hour
CACHE_DURATION_MODEL = 86400  # 24 hours
CACHE_DURATION_PREDICTIONS = 3600  # 1 hour

# ==================== PAGINATION ====================

ROWS_PER_PAGE = int(os.getenv("ROWS_PER_PAGE", "20"))
MAX_ROWS_DISPLAY = 1000

# ==================== DECIMAL PRECISION ====================

DECIMAL_PLACES = int(os.getenv("DECIMAL_PLACES", "4"))
PERCENTAGE_DECIMALS = 2

# ==================== FEATURE IMPORTANCE ====================

# Number of top features to display
TOP_FEATURES_COUNT = 10

# ==================== CROSS VALIDATION ====================

# K-Fold settings
MIN_CV_FOLDS = 3
MAX_CV_FOLDS = 10
DEFAULT_CV_SPLITS = 5

# ==================== HYPERPARAMETER TUNING ====================

# GridSearch vs RandomSearch
GRID_SEARCH_JOBS = -1
RANDOM_SEARCH_N_ITER = 20
RANDOM_SEARCH_CV = 5

# ==================== TIME SETTINGS ====================

# Default datetime format
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'

# ==================== NOTIFICATION SETTINGS ====================

ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() == "true"
NOTIFICATION_TIMEOUT = 5  # seconds

# ==================== SYSTEM SETTINGS ====================

# Memory optimization
MAX_DATASET_SIZE_MB = 100
CHUNK_SIZE = 10000  # rows for processing large datasets

# Parallel processing
N_JOBS = -1  # Use all available processors

print(f"Configuration loaded: {APP_NAME} v{APP_VERSION}")
print(f"Debug mode: {DEBUG}")
print(f"Log level: {LOG_LEVEL}")
