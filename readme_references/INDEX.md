# 📑 PROJECT INDEX - Customer Churn Prediction System

## 🎯 START HERE

### For First-Time Users
1. **Read:** [README.md](README.md) - 5 minutes
2. **Setup:** Run `quickstart.bat` (Windows) or `./quickstart.sh` (Mac/Linux)
3. **Generate Sample Data:** `python generate_sample_data.py`
4. **Login & Explore:** Upload sample_data.csv and try the system

### For Deployment
1. **Read:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. **Choose:** Local, Docker, or Cloud
3. **Follow:** Step-by-step instructions
4. **Deploy & Monitor:** Use provided scripts

### For Development
1. **Read:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. **Review:** Code structure and architecture
3. **Install:** Dependencies from requirements.txt
4. **Explore:** Modular code design
5. **Extend:** Add your own features

---

## 📚 Documentation Map

### Core Documentation
- **[README.md](README.md)** - Complete project overview
  - Features overview
  - Installation guide
  - Basic usage
  - Troubleshooting
  - ~11 KB

- **[USER_GUIDE.md](USER_GUIDE.md)** - User manual
  - Account management
  - Data upload instructions
  - EDA tutorials
  - Model training walkthrough
  - Prediction usage
  - Tips & best practices
  - FAQ section
  - ~14 KB

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Deployment instructions
  - Local development setup
  - Docker deployment
  - Cloud deployment (AWS, GCP, Azure, Heroku)
  - Production checklist
  - Scaling strategies
  - ~8 KB

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical summary
  - Project structure
  - Features implemented
  - Technology stack
  - File inventory
  - Learning outcomes
  - Future enhancements
  - ~13 KB

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick lookup guide
  - 5-minute quick start
  - Navigation guide
  - API reference
  - Troubleshooting table
  - Keyboard shortcuts
  - ~8 KB

### This File
- **[INDEX.md](INDEX.md)** - You are here!
  - Navigation guide
  - File descriptions
  - Quick lookups

---

## 💻 Application Files

### Main Application
- **[app.py](app.py)** - Streamlit main application
  - Entry point for the system
  - Page routing and navigation
  - Authentication checks
  - UI layout and styling
  - ~250 lines

- **[config.py](config.py)** - Application configuration
  - Environment variables
  - Model hyperparameters
  - Constants and settings
  - Color schemes
  - Default values
  - ~200 lines

### Database Layer
- **[database.py](database.py)** - SQLite operations
  - User management (create, authenticate)
  - Dataset tracking
  - Model versioning
  - Prediction logging
  - ~350 lines
  - Tables: users, datasets, models, predictions

### Utilities
- **[utils.py](utils.py)** - Helper functions
  - File operations (save, load, validate)
  - Data analysis (summaries, statistics)
  - Caching utilities
  - Logging setup
  - Progress indicators
  - ~250 lines

### Authentication
- **[src_auth.py](src_auth.py)** - User authentication
  - Login/logout functions
  - Session management
  - Password verification
  - User data retrieval
  - ~120 lines

---

## 🤖 Machine Learning Modules

### Data Preprocessing
- **[preprocessing.py](preprocessing.py)** - Data pipeline
  - DataPreprocessor class
  - Missing value handling (4 methods)
  - Categorical encoding (2 methods)
  - Feature scaling (2 methods)
  - Feature selection
  - Train-test split with stratification
  - Serialization support
  - ~400 lines

### Model Training
- **[train.py](train.py)** - Model management
  - ModelTrainer class (6 algorithms)
  - Logistic Regression, Random Forest, Decision Tree, XGBoost, SVM, KNN
  - Cross-validation methods
  - K-Fold validation
  - ModelHyperparameterTuner class
  - GridSearchCV & RandomizedSearchCV
  - ~450 lines

### Evaluation & Metrics
- **[evaluate.py](evaluate.py)** - Model evaluation
  - ModelEvaluator class
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve & AUC
  - Classification Report
  - Comparison utilities
  - Visualization functions (Matplotlib & Plotly)
  - ~350 lines

### Predictions
- **[predict.py](predict.py)** - Prediction engine
  - PredictionEngine class
  - Single predictions
  - Batch predictions
  - Confidence scoring
  - Risk level assessment
  - BatchPredictionProcessor class
  - Prediction history
  - ~300 lines

---

## 🖥️ Streamlit Pages

### Dashboard Page
- **[page_dashboard.py](page_dashboard.py)**
  - Overview metrics
  - Recent datasets and models
  - Prediction trends
  - Statistics and charts
  - Quick access to features

### Data Upload
- **[upload_page.py](upload_page.py)**
  - CSV file upload
  - File validation
  - Dataset preview
  - Statistics display
  - Requirements checklist

### Data Exploration
- **[page_eda.py](page_eda.py)**
  - Overview tab (statistics, missing values)
  - Distributions tab (histograms, boxplots)
  - Correlations tab (heatmaps)
  - Relationships tab (scatter plots)
  - Outliers tab (detection methods)
  - Target analysis tab

### Model Training
- **[page_training.py](page_training.py)**
  - Preprocessing configuration
  - Model selection
  - Advanced options
  - Training progress
  - Performance comparison

### Predictions
- **[page_prediction.py](page_prediction.py)**
  - Single prediction interface
  - Manual input forms
  - Batch prediction upload
  - Result visualization
  - Risk assessment
  - Recommendations

### Model History
- **[page_history.py](page_history.py)**
  - All trained models list
  - Performance comparison charts
  - Prediction history
  - Model download
  - Trend analysis

### Settings
- **[page_settings.py](page_settings.py)**
  - User profile information
  - Preferences customization
  - Application information
  - Quick tips and tricks

---

## 📦 Configuration Files

### Dependencies
- **[requirements.txt](requirements.txt)**
  - Python package versions
  - ML libraries: scikit-learn, XGBoost, pandas, numpy
  - Web: Streamlit
  - Visualization: Plotly, Matplotlib, Seaborn
  - Database: SQLAlchemy
  - Security: bcrypt
  - Utilities: python-dotenv, Pillow, reportlab

### Environment Variables
- **[.env.example](.env.example)**
  - Database path
  - Application settings
  - Model training defaults
  - Streamlit configuration
  - Security settings
  - Copy and customize for deployment

### Streamlit Config
- **[.streamlit_config.toml](.streamlit_config.toml)**
  - Theme colors
  - Font settings
  - Server configuration
  - Logger level
  - XSRF protection

---

## 🐳 Deployment Files

### Docker
- **[Dockerfile](Dockerfile)**
  - Python 3.10 slim image
  - Dependency installation
  - Directory structure
  - Health checks
  - Port exposure (8501)

- **[docker-compose.yml](docker-compose.yml)**
  - Service configuration
  - Volume mounts
  - Environment variables
  - Health checks
  - Port mapping

### Quick Start Scripts
- **[quickstart.bat](quickstart.bat)** - Windows quick start
  - Virtual environment setup
  - Dependency installation
  - Database initialization
  - Application startup

- **[quickstart.sh](quickstart.sh)** - Linux/Mac quick start
  - Virtual environment setup
  - Dependency installation
  - Database initialization
  - Application startup

---

## 🛠️ Utility Scripts

### Sample Data Generator
- **[generate_sample_data.py](generate_sample_data.py)**
  - Generates realistic customer churn data
  - Synthetic dataset creation
  - Configurable sample size
  - Saved as sample_data.csv
  - Usage: `python generate_sample_data.py`

---

## 📂 Directory Structure

```
project/
├── CODE FILES
├── app.py                    Main application
├── database.py              SQLite operations
├── utils.py                 Utility functions
├── config.py                Configuration
├── src_auth.py              Authentication
├── preprocessing.py         Data preprocessing
├── train.py                 Model training
├── evaluate.py              Model evaluation
├── predict.py               Predictions
├── page_dashboard.py        Dashboard page
├── page_eda.py              EDA page
├── page_training.py         Training page
├── page_prediction.py       Prediction page
├── page_history.py          History page
├── page_settings.py         Settings page
├── upload_page.py           Upload page
│
├── CONFIGURATION
├── requirements.txt         Dependencies
├── .env.example             Environment template
├── .streamlit_config.toml   Streamlit config
├── config.py                Python config
│
├── DEPLOYMENT
├── Dockerfile               Docker image
├── docker-compose.yml       Docker compose
├── quickstart.bat           Windows startup
├── quickstart.sh            Linux/Mac startup
│
├── DOCUMENTATION
├── README.md                Main documentation
├── USER_GUIDE.md            User manual
├── DEPLOYMENT_GUIDE.md      Deployment instructions
├── PROJECT_SUMMARY.md       Technical summary
├── QUICK_REFERENCE.md       Quick lookup
├── INDEX.md                 This file
│
├── UTILITIES
├── generate_sample_data.py  Sample data generator
├── pages_init.txt           Pages marker
│
├── DATA DIRECTORIES (created at runtime)
├── database/                SQLite storage
├── datasets/                Uploaded CSVs
├── models/                  Trained models
└── logs/                    Application logs
```

---

## 🎯 Feature Matrix

| Feature | Module | File | Status |
|---------|--------|------|--------|
| Authentication | auth | src_auth.py | ✅ Complete |
| Data Upload | upload | upload_page.py | ✅ Complete |
| EDA | analysis | page_eda.py | ✅ Complete |
| Preprocessing | pipeline | preprocessing.py | ✅ Complete |
| Model Training | training | train.py, page_training.py | ✅ Complete |
| Evaluation | metrics | evaluate.py | ✅ Complete |
| Predictions | inference | predict.py, page_prediction.py | ✅ Complete |
| Hyperparameter Tuning | optimization | train.py | ✅ Complete |
| Model Monitoring | tracking | database.py, page_history.py | ✅ Complete |
| Dashboard | UI | page_dashboard.py | ✅ Complete |

---

## 🚀 Quick Navigation by Task

### I want to...

#### Train a Model
1. Go to: [page_training.py](page_training.py)
2. Uses: [preprocessing.py](preprocessing.py), [train.py](train.py)
3. Stores: [database.py](database.py)
4. Guide: [USER_GUIDE.md](USER_GUIDE.md) - "Model Training" section

#### Make Predictions
1. Go to: [page_prediction.py](page_prediction.py)
2. Uses: [predict.py](predict.py), [database.py](database.py)
3. Guide: [USER_GUIDE.md](USER_GUIDE.md) - "Making Predictions" section

#### Explore Data
1. Go to: [page_eda.py](page_eda.py)
2. Uses: [utils.py](utils.py)
3. Guide: [USER_GUIDE.md](USER_GUIDE.md) - "EDA" section

#### Deploy Application
1. Read: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Use: [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml)
3. Or: [quickstart.bat](quickstart.bat) / [quickstart.sh](quickstart.sh)

#### Understand Architecture
1. Read: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Review: Code structure in this document
3. Explore: [README.md](README.md)

#### Find Quick Answers
1. Check: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Search: [INDEX.md](INDEX.md) (this file)
3. Ask: [USER_GUIDE.md](USER_GUIDE.md) - "FAQ" section

---

## 📊 Code Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 17 |
| Total Lines of Code | 6000+ |
| Documentation Files | 6 |
| Configuration Files | 4 |
| ML Algorithms | 6 |
| Evaluation Metrics | 5+ |
| Supported Data Formats | 1 (CSV) |

---

## ✨ Highlighted Features

**Best For Learning:**
- Well-commented code
- Modular architecture
- Clear separation of concerns
- Object-oriented design

**Best For Production:**
- Error handling
- Input validation
- Logging system
- Database persistence
- Docker deployment

**Best For Customization:**
- Configuration file
- Modular design
- Environment variables
- Extensible classes

---

## 🔗 File Dependencies

```
app.py
├── src_auth.py
├── database.py
├── utils.py
├── config.py
└── page_*.py
    ├── database.py
    ├── utils.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

---

## 💡 Pro Tips

1. **First Time?** Start with [README.md](README.md)
2. **Need Help?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. **Deploy?** Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
4. **Using System?** Read [USER_GUIDE.md](USER_GUIDE.md)
5. **Want Details?** See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## 📞 Support Resources

| Need | Resource | Location |
|------|----------|----------|
| Getting Started | README.md | Root |
| How to Use | USER_GUIDE.md | Root |
| Deployment | DEPLOYMENT_GUIDE.md | Root |
| Quick Answers | QUICK_REFERENCE.md | Root |
| Technical Details | PROJECT_SUMMARY.md | Root |
| Code Examples | Docstrings | Each file |
| Configuration | config.py | Root |

---

## 🎓 Learning Path

### Beginner
1. Read README.md
2. Run quickstart
3. Generate sample data
4. Try system with sample data
5. Read USER_GUIDE.md

### Intermediate
1. Read PROJECT_SUMMARY.md
2. Review code structure
3. Train custom models
4. Deploy locally

### Advanced
1. Read DEPLOYMENT_GUIDE.md
2. Deploy to cloud
3. Customize code
4. Extend features

---

## ✅ Checklist

Before starting:
- [ ] Read this INDEX
- [ ] Check README.md
- [ ] Run quickstart script
- [ ] Generate sample data
- [ ] Create account
- [ ] Upload sample data
- [ ] Explore features
- [ ] Train a model
- [ ] Make predictions

---

## 📝 File Purpose Summary

| File | Purpose | Priority |
|------|---------|----------|
| app.py | Main entry point | 🔴 Critical |
| database.py | Data persistence | 🔴 Critical |
| preprocessing.py | Data pipeline | 🔴 Critical |
| train.py | Model training | 🔴 Critical |
| evaluate.py | Model evaluation | 🟠 Important |
| predict.py | Predictions | 🟠 Important |
| config.py | Configuration | 🟠 Important |
| Page files | UI/UX | 🟠 Important |
| utils.py | Helpers | 🟡 Supporting |
| src_auth.py | Authentication | 🟡 Supporting |

---

## 🏁 Next Steps

1. **Start Here**: Open [README.md](README.md)
2. **Setup**: Run quickstart script
3. **Learn**: Read [USER_GUIDE.md](USER_GUIDE.md)
4. **Explore**: Use system features
5. **Deploy**: Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Welcome to Customer Churn Prediction System!**

**Version:** 1.0.0  
**Status:** Production Ready  
**Last Updated:** 2024

For questions, refer to the appropriate documentation file above.
