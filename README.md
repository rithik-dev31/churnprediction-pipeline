# Customer Churn Prediction System

A comprehensive, production-ready Machine Learning pipeline for predicting customer churn using multiple algorithms, advanced preprocessing techniques, and an interactive Streamlit dashboard.

## 🌟 Features

### Authentication & Security
- ✅ Secure user authentication with bcrypt password hashing
- ✅ Session management with Streamlit state
- ✅ User registration and login pages
- ✅ Input validation and error handling

### Data Management
- ✅ CSV dataset upload and storage
- ✅ Automated data validation
- ✅ Dataset summary and statistics
- ✅ Shape and column type information
- ✅ Missing value and duplicate detection

### Exploratory Data Analysis (EDA)
- ✅ Dataset overview and statistics
- ✅ Feature distributions (histograms, boxplots)
- ✅ Categorical analysis (countplots, pie charts)
- ✅ Correlation heatmaps
- ✅ Pairplot analysis
- ✅ Outlier detection (IQR and Z-score methods)
- ✅ Target variable analysis
- ✅ Interactive visualizations with Plotly

### Data Preprocessing
- ✅ Missing value handling (mean, median, drop, forward fill)
- ✅ Categorical encoding (Label, One-Hot)
- ✅ Feature scaling (StandardScaler, MinMaxScaler)
- ✅ Feature selection (SelectKBest)
- ✅ Train-test stratified split
- ✅ sklearn Pipeline architecture
- ✅ Preprocessor serialization with joblib

### Machine Learning Models
**6 state-of-the-art algorithms:**
1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Robust ensemble method
3. **Decision Tree** - Easy to understand
4. **XGBoost** - High-performance gradient boosting
5. **Support Vector Machine** - Powerful non-linear classifier
6. **K-Nearest Neighbors** - Simple and effective

### Model Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion Matrix visualization
- ✅ ROC Curve and AUC score
- ✅ Classification Report
- ✅ Cross-Validation (K-Fold)
- ✅ Model comparison charts
- ✅ Performance tracking over time

### Hyperparameter Tuning
- ✅ GridSearchCV for exhaustive search
- ✅ RandomizedSearchCV for efficient exploration
- ✅ Customizable parameter grids per model
- ✅ Cross-validation integration

### Predictions
- ✅ Single prediction interface with manual input
- ✅ Batch prediction from CSV files
- ✅ Prediction confidence scores
- ✅ Risk level assessment
- ✅ Personalized recommendations
- ✅ Prediction history and logging
- ✅ Batch result export to CSV

### Dashboard & Monitoring
- ✅ Real-time performance metrics
- ✅ Dataset and model statistics
- ✅ Prediction trends visualization
- ✅ Model history and versioning
- ✅ Recent predictions display
- ✅ Churn/Retain distribution charts

### Database
- ✅ SQLite for data persistence
- ✅ User management
- ✅ Dataset tracking
- ✅ Model versioning
- ✅ Prediction logging

### Deployment Ready
- ✅ requirements.txt with all dependencies
- ✅ Docker support (Dockerfile included)
- ✅ Streamlit configuration file
- ✅ Environment variable support
- ✅ Logging system
- ✅ Error handling and validation

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28+
- scikit-learn 1.3+
- XGBoost 2.0+
- Pandas 2.1+
- NumPy 1.24+
- Plotly 5.18+
- Matplotlib 3.8+
- Seaborn 0.13+

## 🚀 Installation

### 1. Clone or Download the Repository
```bash
cd customer-churn-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python -c "from database import init_database; init_database()"
```

### 5. Run Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📖 Usage Guide

### Getting Started

1. **Create Account**
   - Click "Sign Up" in the sidebar
   - Enter username, email, and password
   - Click "Sign Up"

2. **Login**
   - Enter credentials
   - Click "Login"

3. **Upload Dataset**
   - Go to "Upload Data" page
   - Upload a CSV file with customer data
   - Ensure it includes a 'Churn' column (0 or 1)
   - Minimum 10 rows required

4. **Explore Data**
   - Go to "Explore Data" page
   - Select your dataset
   - View statistics, distributions, correlations
   - Analyze feature relationships and outliers
   - Study target variable distribution

5. **Train Models**
   - Go to "Train Models" page
   - Select dataset and preprocessing options
   - Choose models to train
   - Configure advanced options
   - Click "Start Training"
   - Review performance metrics

6. **Make Predictions**
   - Go to "Make Predictions" page
   - Select a trained model
   - Enter customer information (single) or upload CSV (batch)
   - Get churn prediction and confidence score
   - Review recommendations

7. **Monitor Performance**
   - Go to "Model History" page
   - View all trained models and their metrics
   - Compare model performance
   - Track prediction history
   - Download models

## 📊 Dataset Format

Your dataset should have the following structure:

```
CustomerID | Age | Tenure | MonthlyCharges | Churn
-----------|-----|--------|-----------------|------
1          | 32  | 24     | 65.5           | 0
2          | 45  | 12     | 89.2           | 1
3          | 28  | 36     | 45.0           | 0
```

**Required:**
- Target variable column named 'Churn' (values: 0 or 1)
- At least 10 rows
- At least 2 columns
- No completely empty columns

**Supported data types:**
- Numerical (int, float)
- Categorical (object/string)

## 🏗️ Project Structure

```
project/
├── app.py                  # Main Streamlit application
├── database.py             # SQLite database operations
├── utils.py                # Utility functions
├── preprocessing.py        # Data preprocessing pipeline
├── train.py                # Model training
├── evaluate.py             # Model evaluation metrics
├── predict.py              # Prediction engine
├── src_auth.py             # Authentication module
│
├── page_dashboard.py       # Dashboard page
├── page_eda.py             # EDA analysis page
├── page_training.py        # Model training page
├── page_prediction.py      # Prediction page
├── page_history.py         # Model history page
├── page_settings.py        # Settings page
├── upload_page.py          # Data upload page
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # Environment variables template
├── Dockerfile             # Docker configuration
│
├── database/              # SQLite database storage
├── datasets/              # Uploaded CSV files
├── models/                # Trained model files
└── logs/                  # Application logs
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file based on `.env.example`:

```
# Database
DATABASE_PATH=database/churn_prediction.db

# Application
APP_NAME=Customer Churn Prediction
DEBUG=False
LOG_LEVEL=INFO

# Model Training
DEFAULT_TEST_SIZE=0.2
DEFAULT_CV_FOLDS=5
DEFAULT_RANDOM_STATE=42

# Streamlit
MAX_UPLOAD_SIZE=200
```

### Streamlit Configuration
Modify `.streamlit/config.toml` to customize:
- Theme colors
- Page layout
- Server settings
- Logger level

## 📈 Model Performance

### Evaluation Metrics

**Accuracy**
- Proportion of correct predictions
- Use when classes are balanced

**Precision**
- Proportion of positive predictions that are correct
- Important when false positives are costly

**Recall**
- Proportion of actual positives correctly identified
- Important when false negatives are costly

**F1-Score**
- Harmonic mean of precision and recall
- Balanced measure for imbalanced datasets

**ROC-AUC**
- Area under the Receiver Operating Characteristic curve
- Evaluates model across all classification thresholds

## 🎯 Typical Workflow

1. **Data Upload** → Validate data format
2. **EDA** → Understand data characteristics
3. **Preprocessing** → Handle missing values, scale features
4. **Model Training** → Train multiple models
5. **Evaluation** → Compare model performance
6. **Hyperparameter Tuning** → Optimize best model
7. **Predictions** → Make predictions on new data
8. **Monitoring** → Track model performance over time

## 🔐 Security Considerations

- ✅ Passwords hashed with bcrypt
- ✅ Session-based authentication
- ✅ Input validation on all forms
- ✅ File upload validation
- ✅ SQL injection prevention with parameterized queries
- ✅ Sensitive data isolation per user

## 📝 Logging

Application logs are stored in `logs/` directory:
- `app.log` - General application logs
- `events.log` - User events and actions

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t churn-prediction:latest .
```

### Run Container
```bash
docker run -p 8501:8501 -v $(pwd)/database:/app/database churn-prediction:latest
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./database:/app/database
      - ./datasets:/app/datasets
      - ./models:/app/models
```

## 🌐 Deployment Guide

### Moving this project to GitHub

Follow these commands in your project root (replace <username>/<repo>):

1. Initialize and commit:

```
 git init
 git branch -M main
 git add .
 git commit -m "Initial commit"
```

2. Create a GitHub repo (website) or use GitHub CLI:

```
 gh repo create <username>/<repo> --public --source=. --remote=origin --push
```

3. Or add remote and push manually:

```
 git remote add origin https://github.com/<username>/<repo>.git
 git push -u origin main
```

Then continue with Streamlit Cloud steps:

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect GitHub repository
4. Deploy directly

### AWS EC2 Deployment
1. Launch EC2 instance with Python
2. Clone repository
3. Install dependencies
4. Run: `streamlit run app.py --server.address=0.0.0.0`

### Heroku Deployment
1. Create `Procfile`: `web: streamlit run app.py`
2. Create `.streamlit/config.toml`
3. Deploy using Heroku CLI

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Database errors
**Solution:** Reinitialize database
```bash
rm database/churn_prediction.db
python -c "from database import init_database; init_database()"
```

### Issue: Large file uploads fail
**Solution:** Increase upload limit in `.streamlit/config.toml`
```
[client]
maxUploadSize = 500
```

### Issue: Model predictions are slow
**Solution:** Consider reducing model complexity or using fewer features

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Plotly Documentation](https://plotly.com/python)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Developed by ML Engineering Team

## 📞 Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Last Updated:** 2026
**Version:** 1.0.0
