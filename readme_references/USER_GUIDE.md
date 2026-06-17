# User Guide - Customer Churn Prediction System

## Table of Contents
1. [Getting Started](#getting-started)
2. [Account Management](#account-management)
3. [Data Management](#data-management)
4. [EDA - Exploratory Data Analysis](#eda---exploratory-data-analysis)
5. [Model Training](#model-training)
6. [Making Predictions](#making-predictions)
7. [Model Monitoring](#model-monitoring)
8. [Tips & Best Practices](#tips--best-practices)
9. [FAQ](#faq)

---

## Getting Started

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Stable internet connection
- CSV dataset (customer data with churn indicator)

### First Login
1. Open application at `http://localhost:8501`
2. Click **"Sign Up"** in the sidebar
3. Enter:
   - Username (3+ characters)
   - Email address
   - Password (6+ characters)
4. Confirm password and click **"Sign Up"**
5. Login with your credentials

### Main Dashboard
After login, you'll see:
- **Key Metrics**: Datasets, Models, Predictions, Average Accuracy
- **Recent Datasets**: Your uploaded datasets
- **Best Models**: Top performing trained models
- **Recent Predictions**: Latest predictions made
- **Statistics**: Visual breakdown of predictions

---

## Account Management

### Changing Settings
1. Click **"Settings"** in sidebar
2. Three tabs available:
   - **Profile**: View your username, email, account details
   - **Preferences**: Customize theme, notifications, display options
   - **About**: Application information and quick tips

### Logging Out
Click **🚪 Logout** button in sidebar
- Session ends immediately
- Must login again to access features

### Password Security
- Always use strong passwords (8+ characters with mix of letters, numbers)
- Never share your credentials
- Passwords are encrypted with bcrypt

---

## Data Management

### Uploading Data

1. **Navigate to Upload Page**
   - Click "Upload Data" in sidebar

2. **Select CSV File**
   - Click upload area
   - Choose your CSV file
   - File preview will display

3. **Review Dataset Information**
   - **Rows**: Number of samples
   - **Columns**: Number of features
   - **File Size**: Size in KB
   - **Column Info**: Data types and missing values
   - **Statistical Summary**: Basic statistics

4. **Confirm Upload**
   - Click **"📤 Upload"** button
   - Dataset is saved to your account
   - Ready for analysis

### Dataset Requirements

**Must Have:**
- ✅ CSV format (.csv extension)
- ✅ Column named 'Churn' (target variable)
- ✅ Minimum 10 rows
- ✅ At least 2 columns
- ✅ No completely empty columns

**Format Example:**
```
CustomerID | Age | Tenure | MonthlyCharges | Churn
1          | 32  | 24     | 65.5           | 0
2          | 45  | 12     | 89.2           | 1
3          | 28  | 36     | 45.0           | 0
```

**Target Variable (Churn):**
- 0 = Customer will NOT churn (retain)
- 1 = Customer WILL churn (leave)

### Managing Datasets

1. View all your datasets in **"Upload Data"** page
2. See:
   - File size
   - Number of rows and columns
   - Upload date

---

## EDA - Exploratory Data Analysis

### Why EDA Matters
- Understand data characteristics
- Identify patterns and trends
- Detect anomalies and outliers
- Inform preprocessing decisions

### EDA Tabs

#### 1. Overview Tab
- **Dataset Statistics**: Shape, memory usage, duplicates
- **Data Types**: Column names, types, missing values
- **Missing Values Table**: Count and percentage per column

#### 2. Distributions Tab
- **Numerical Features**: Histograms and box plots
- **Statistical Summary**: Min, max, mean, std, quartiles
- **Categorical Features**: Bar charts and value counts

#### 3. Correlations Tab
- **Heatmap**: Show relationships between numerical features
- **High Correlations**: List feature pairs with correlation > 0.7
- **Interpretation**: Red = positive, blue = negative correlation

#### 4. Relationships Tab
- **Scatter Plots**: Visualize relationships between two features
- **Trend Analysis**: Identify linear or non-linear patterns
- **Outlier Detection**: Spot unusual data points

#### 5. Outliers Tab
- **IQR Method**: Identifies outliers using interquartile range
- **Z-Score Method**: Identifies outliers using standard deviations
- **Statistics**: Shows count and percentage of outliers

#### 6. Target Analysis Tab
- **Distribution**: Pie chart of churn vs no-churn
- **Statistics**: Count and percentages
- **Class Balance**: Check if classes are balanced

### Interpreting Results

**Good Signs:**
- ✅ No missing values
- ✅ No duplicates
- ✅ Balanced target variable (roughly 40-60%)
- ✅ Features show variation
- ✅ Few extreme outliers

**Areas to Address:**
- ⚠️ Many missing values → Need imputation
- ⚠️ Imbalanced classes → Consider class weights
- ⚠️ Highly correlated features → May remove redundant ones
- ⚠️ Extreme outliers → May affect model

---

## Model Training

### Training Workflow

1. **Select Dataset**
   - Choose from your uploaded datasets
   - System verifies it contains 'Churn' column

2. **Configure Preprocessing**
   - **Missing Values**: mean, median, or drop
   - **Encoding**: label (ordinal) or onehot (binary)
   - **Scaling**: standard (mean=0) or minmax (0-1)
   - **Test Size**: Usually 20% for testing

3. **Select Models**
   - Choose which algorithms to train
   - Can select all or subset
   - Each model is independent

4. **Advanced Options**
   - **Feature Selection**: Reduce to top K features
   - **Cross-Validation**: K-Fold validation (typically 5)

5. **Start Training**
   - Click **"🚀 Start Training"**
   - Progress bar shows status
   - May take seconds to minutes

### Understanding Metrics

#### Accuracy
- **Definition**: Proportion of correct predictions
- **Range**: 0-100%
- **When to use**: Balanced datasets
- **Interpretation**: 
  - 80% accuracy = 4 correct out of 5

#### Precision
- **Definition**: Of predicted churners, how many actually churned?
- **Formula**: True Positives / (TP + FP)
- **When to use**: When false positives are costly
- **Interpretation**: 
  - 90% precision = of 10 predicted churners, 9 were correct

#### Recall
- **Definition**: Of actual churners, how many did we catch?
- **Formula**: True Positives / (TP + FN)
- **When to use**: When missing churners is costly
- **Interpretation**:
  - 85% recall = we caught 85% of actual churners

#### F1-Score
- **Definition**: Balanced measure of precision and recall
- **Range**: 0-1 (higher is better)
- **When to use**: Best overall model evaluation
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)

### Model Comparison

After training:
1. View metrics table comparing all models
2. Bar chart shows visual comparison
3. Best model highlighted with highest F1-Score

### Choosing Best Model

**Considerations:**
- F1-Score: Best balanced metric
- Accuracy: If dataset is balanced
- Precision: If false positives are costly
- Recall: If false negatives are costly

**Example Scenarios:**
- Telecom churn: Focus on **Recall** (don't miss churners)
- Credit card: Focus on **Precision** (minimize false alarms)
- General: Use **F1-Score** (balanced approach)

---

## Making Predictions

### Single Prediction

1. **Navigate to "Make Predictions"**
   - Select a trained model
   - Provide dataset reference

2. **Enter Customer Information**
   - Fill in all features
   - Numerical fields: sliders or input
   - Categorical fields: dropdown select

3. **Get Prediction**
   - Click **"🔮 Make Prediction"**
   - Receive:
     - **Prediction**: Churn (🔴) or Retain (🟢)
     - **Confidence**: 0-100% certainty
     - **Risk Level**: Very High to Very Low

4. **View Recommendation**
   - System provides specific actions
   - Based on prediction outcome

### Batch Prediction

1. **Select "Batch Prediction"**
   - Upload CSV with customer data
   - Same features as training data

2. **Process Predictions**
   - Click **"🔮 Make Batch Predictions"**
   - System processes all rows

3. **Review Results**
   - See all predictions in table
   - Statistics: Total, Churn count, Retain count
   - Average confidence level

4. **Download Results**
   - Click **"📥 Download Results"**
   - CSV file with all predictions
   - Use for further analysis

### Interpretation Guide

**Risk Level Scale:**
- Very High: 90-100% confidence
- High: 80-90% confidence
- Medium: 70-80% confidence
- Low: 60-70% confidence
- Very Low: Below 60% confidence

**Action Recommendations:**

**If Churn Predicted (🔴):**
- 📞 Proactive outreach call
- 💰 Special retention offer
- 🎁 Loyalty rewards
- 👥 Dedicated support
- 📊 Analyze pain points
- 🔄 Service improvement plan

**If Retain Predicted (🟢):**
- 📈 Upsell opportunities
- 🎁 Loyalty benefits
- 📧 Regular communication
- 👍 Maintain quality
- 🎯 Expand services

---

## Model Monitoring

### Accessing Model History

1. **Click "Model History"** in sidebar
2. Three tabs available:
   - All Models
   - Performance Comparison
   - Prediction History

### All Models Tab

Shows each trained model with:
- **Name & Type**: Model identifier
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Created Date**: When model was trained
- **Download Button**: Save model locally

### Performance Comparison Tab

- **Table View**: Compare all metrics across models
- **Chart View**: Visual comparison of metrics
- **Best Model**: Top performer highlighted

### Prediction History Tab

- **Statistics**: Total, Churn, Retain, Avg Confidence
- **Recent Predictions**: List of latest predictions
- **Trend Chart**: Predictions over time
- **Download History**: Export all predictions

### Model Versioning

System keeps all trained models:
- Multiple versions from different datasets
- Different preprocessing strategies
- Compare performance over time
- Revert to older versions if needed

---

## Tips & Best Practices

### Data Preparation
1. ✅ Clean data before uploading
2. ✅ Remove duplicate rows
3. ✅ Ensure target variable is binary (0/1)
4. ✅ Handle missing values appropriately
5. ✅ Check for data quality issues

### Model Training
1. ✅ Try multiple models
2. ✅ Use cross-validation
3. ✅ Don't overfit (test on separate data)
4. ✅ Balance performance metrics
5. ✅ Document model versions
6. ✅ Retrain periodically with new data

### Prediction Best Practices
1. ✅ Verify input data completeness
2. ✅ Use model with best F1-Score
3. ✅ Consider confidence levels
4. ✅ Don't rely solely on automated predictions
5. ✅ Combine with business knowledge
6. ✅ Monitor prediction accuracy

### Feature Engineering
1. ✅ Create meaningful features
2. ✅ Remove highly correlated features
3. ✅ Encode categorical properly
4. ✅ Scale numerical features
5. ✅ Select important features

### Model Evaluation
1. ✅ Always use test set
2. ✅ Check multiple metrics
3. ✅ Perform cross-validation
4. ✅ Compare model performance
5. ✅ Analyze misclassifications

---

## FAQ

### Q: How do I improve model accuracy?
**A:** 
- Get more data (larger dataset)
- Better feature engineering
- Hyperparameter tuning
- Try different algorithms
- Handle class imbalance
- Remove noisy features

### Q: What's the difference between Precision and Recall?
**A:**
- **Precision**: Accuracy of positive predictions (avoid false alarms)
- **Recall**: Coverage of positive cases (avoid missing them)
- Use together as F1-Score for balance

### Q: Can I use categorical features?
**A:** Yes! System handles encoding automatically:
- Label encoding: Ordinal categories
- One-hot encoding: Nominal categories

### Q: How often should I retrain models?
**A:** Recommendations:
- Weekly: If data changes rapidly
- Monthly: For typical business scenarios
- Quarterly: For stable patterns
- When accuracy drops significantly

### Q: What if my dataset is imbalanced (80% No Churn, 20% Churn)?
**A:**
- System handles this automatically
- Use F1-Score instead of Accuracy
- Consider class weights
- Oversample minority class
- Undersample majority class

### Q: Can I export models?
**A:** Yes!
- Go to "Model History"
- Click download button next to model
- Save .joblib file locally
- Load later using joblib.load()

### Q: What if predictions seem wrong?
**A:**
- Check model confidence level
- Review prediction history
- Retrain with more data
- Try different preprocessing
- Combine with domain expertise

### Q: How do I interpret the confidence score?
**A:**
- 0.9 (90%): Very confident, high reliability
- 0.7 (70%): Reasonably confident
- 0.5 (50%): Uncertain, close to random
- Low confidence? Consider ensemble predictions

### Q: Can I use with production data?
**A:**
- Yes, but test first
- Validate on sample before batch
- Monitor predictions over time
- Have fallback strategies
- Regularly update models

### Q: How is my data secured?
**A:**
- Per-user database isolation
- Encrypted password storage
- Local database (SQLite)
- No external data sharing
- User-controlled deletion

### Q: What if I run out of storage?
**A:**
- Archive old models: `rm models/old_model.joblib`
- Clean datasets: Remove unused uploaded files
- Check disk space: Different by OS
- Implement database cleanup

---

## Getting Help

### Resources
- **README.md**: Technical overview
- **DEPLOYMENT_GUIDE.md**: Deployment instructions
- **config.py**: Configuration reference
- **Documentation**: Inline code comments

### Common Issues
1. **Login fails**: Check username/password, reset database
2. **Upload fails**: Verify CSV format and requirements
3. **Training slow**: Reduce dataset size, use fewer models
4. **Predictions incorrect**: Retrain with better data

### Support Channels
- GitHub Issues: Report bugs
- Documentation: Read guides
- Code comments: Technical details
- Logs: Debug information (logs/ directory)

---

**Last Updated:** 2024
**Version:** 1.0.0

For more information, see README.md and DEPLOYMENT_GUIDE.md
