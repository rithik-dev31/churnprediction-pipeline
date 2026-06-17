# ⚡ STREAMLIT - QUICK START (2 Minutes)

## The 3-Command Setup 🚀

### Command 1: Install
```bash
pip install -r requirements.txt
```

### Command 2: Initialize (First Time Only)
```bash
python -c "from database import init_database; init_database()"
```

### Command 3: Run
```bash
streamlit run app.py
```

**Browser opens automatically! 🎉**

---

## What You See 🎨

```
TERMINAL OUTPUT:
═══════════════════════════════════════════════════
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
═══════════════════════════════════════════════════

BROWSER:
═══════════════════════════════════════════════════
http://localhost:8501

┌─────────────────────────────────────────┐
│  📊 Customer Churn Prediction System     │
│                                         │
│  Welcome! 👋                            │
│  This is a comprehensive ML pipeline    │
│                                         │
│  Features:                              │
│  • 🔐 Secure authentication             │
│  • 📈 Interactive EDA                   │
│  • 🤖 6 ML models                       │
│  • 🎯 Real-time predictions             │
│                                         │
│              [Start Using →]            │
└─────────────────────────────────────────┘

LEFT SIDEBAR:
┌─────────────────┐
│ Choose Mode:    │
│ • Login         │
│ • Signup        │
└─────────────────┘
```

---

## Usage Flow 📋

```
START
  ↓
Sign Up (First time only)
  ├─ Username: your_name
  ├─ Email: your@email.com
  ├─ Password: ••••••••
  └─ Confirm: ••••••••
  ↓
Login
  ├─ Username: your_name
  └─ Password: ••••••••
  ↓
MAIN DASHBOARD (After login)
  ├─ Dashboard .................. See stats
  ├─ Upload Data ................ Upload CSV ⭐
  ├─ Explore Data ............... View charts
  ├─ Train Models ............... Train ML ⭐
  ├─ Make Predictions ........... Predict ⭐
  ├─ Model History .............. View past models
  └─ Settings ................... Preferences
```

---

## Main Pages 📑

### 1️⃣ Dashboard
```
Shows:
✓ Total datasets uploaded
✓ Total models trained
✓ Best model accuracy
✓ Recent predictions
```

### 2️⃣ Upload Data ⭐ START HERE
```
1. Click "Upload Data" from sidebar
2. Select your CSV file
3. Click "Upload"
4. See: "Dataset uploaded successfully!"
```

### 3️⃣ Explore Data
```
Shows:
✓ Age distribution chart
✓ Credit score distribution
✓ Correlation heatmap
✓ Missing values info
✓ Statistical summary
```

### 4️⃣ Train Models ⭐ MAIN ACTION
```
1. Select dataset
2. Configure preprocessing
3. Select models (6 available)
4. Click "Train Models"
5. See results + metrics
```

### 5️⃣ Make Predictions ⭐ USE MODEL
```
1. Select trained model
2. Input customer data
3. Click "Predict"
4. Get probability + recommendation
```

### 6️⃣ Model History
```
Shows:
✓ All trained models
✓ Training dates
✓ Model accuracy
✓ Load/Download options
```

### 7️⃣ Settings
```
Change:
✓ Username
✓ Email
✓ Password
✓ Theme (Light/Dark)
✓ Default model
```

---

## Step-by-Step Guide 👣

### Step 1: Start Streamlit
```bash
streamlit run app.py
```
→ Browser opens automatically

### Step 2: Sign Up (First Time)
In sidebar, click "Signup":
```
Username: john_doe
Email: john@example.com
Password: mypassword123
Confirm: mypassword123

[Sign Up Button]
```
→ See: "Account created! Please login."

### Step 3: Login
In sidebar, click "Login":
```
Username: john_doe
Password: mypassword123

[Login Button]
```
→ See: "Login successful!" ✓

### Step 4: Upload Your Data
Click "Upload Data":
```
📁 Choose File → Select churn_data.csv
[Upload Button]
```
→ See: "Dataset uploaded successfully!"

### Step 5: Explore Data
Click "Explore Data":
```
View:
📈 Age Distribution (chart)
📊 Credit Score Distribution (chart)
🔗 Correlation Matrix (heatmap)
📋 Data Summary (statistics)
```

### Step 6: Train Models
Click "Train Models":
```
Select Dataset:
└─ churn_data.csv (10000 rows) ✓

Preprocessing:
├─ Missing Values: mean ✓
├─ Encoding: label ✓
├─ Scaling: standard ✓
└─ Feature Selection: None ✓

Models to Train:
☑ Logistic Regression
☑ Random Forest
☑ Decision Tree
☑ XGBoost
☑ SVM
☑ KNN

[TRAIN MODELS Button]
```

Results show:
```
✓ Logistic Regression: 85.2% accuracy
✓ Random Forest: 87.5% accuracy
✓ Decision Tree: 84.3% accuracy
✓ XGBoost: 89.2% accuracy ⭐ BEST
✓ SVM: 86.7% accuracy
✓ KNN: 83.1% accuracy

Detailed Metrics for XGBoost:
├─ Accuracy: 0.892
├─ Precision: 0.823
├─ Recall: 0.756
├─ F1 Score: 0.788
└─ AUC-ROC: 0.931
```

### Step 7: Make Predictions
Click "Make Predictions":
```
Select Model: XGBoost (Best) ✓

Input Data:
Age: 42
CreditScore: 619
Geography: France
Gender: Female
Balance: 0
Tenure: 2
NumOfProducts: 1
HasCrCard: 1
IsActiveMember: 1
EstimatedSalary: 101348

[PREDICT Button]
```

Result:
```
🎯 PREDICTION RESULT

Will Churn? NO (Class 0)
Confidence: 78.5%

Probabilities:
├─ Will NOT churn: 78.5% ✓ (Selected)
└─ Will churn: 21.5%
```

### Step 8: Check History
Click "Model History":
```
Model                    Accuracy   Date        Status
─────────────────────────────────────────────────────
XGBoost                  89.2%      2 hours ago Active ✓
Random Forest            87.5%      today       Saved
Logistic Regression      85.2%      yesterday   Saved
SVM                      86.7%      2 days ago  Saved

[Load] [View Details] [Delete] [Export]
```

---

## Keyboard Shortcuts ⌨️

| Shortcut | Action |
|----------|--------|
| `Ctrl + C` | Stop Streamlit |
| `Ctrl + Z` | Undo upload |
| `Ctrl + R` | Refresh page |

---

## Troubleshooting 🔧

### Problem: "Address already in use"
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

### Problem: "Database error"
```bash
# Solution: Reinitialize database
python -c "from database import init_database; init_database()"
```

### Problem: "Module not found"
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

### Problem: Slow performance
```bash
# Solution: Close browser tabs, restart Streamlit
streamlit run app.py --logger.level=warning
```

---

## Tips 💡

✅ **First time?**
- Sign up with email
- Upload sample data
- Explore to understand data
- Train models
- Make predictions

✅ **Multiple datasets?**
- Upload each separately
- Switch between them
- Train different models per dataset

✅ **Save models?**
- All models auto-saved
- Check "Model History"
- Download for later use

✅ **Share access?**
- Use "Network URL"
- Share with others on same WiFi
- Each person needs account

---

## Complete Example Workflow 🎯

```
1. Install & Run
   pip install -r requirements.txt
   streamlit run app.py

2. Sign Up (First time)
   Username: demo_user
   Password: demo123

3. Login
   Username: demo_user
   Password: demo123

4. Upload Data
   Select: churn_data.csv (10,000 rows)

5. Explore
   View: Age, Credit Score, Balance distributions

6. Train Models
   Select: All 6 models
   Result: XGBoost best (89.2% accuracy)

7. Predict
   Input: New customer data
   Output: Will NOT churn (78.5% confidence)

8. Save
   Check Model History
   Download best model if needed

DONE! 🎉
```

---

## File Size Limits 📦

```
Max upload size: 200 MB per file
Max model size: 500 MB per model
Session timeout: 24 hours
```

---

## Performance Tips 🚀

```
For faster performance:
• Use dataset < 100,000 rows
• Limit to 3-4 models at once
• Enable feature selection (k=10)
• Close other browser tabs
• Use latest Chrome/Firefox
```

---

## When to Use 💻

| Task | Tool |
|------|------|
| Quick pipeline run | `python run_full_pipeline.py` |
| Interactive training | `streamlit run app.py` ← You are here! |
| Batch predictions | Use Python script |
| Production deployment | Docker container |

---

## What Next? 🎯

After training models in Streamlit:

1. ✅ Review performance in dashboard
2. ✅ Make predictions on new data
3. ✅ Download best model
4. ✅ Use in your application
5. ✅ Deploy to production

---

## Quick Reference 📝

```
COMMAND           PURPOSE
────────────────────────────────────
streamlit run app.py    Start web app
Ctrl + C                Stop web app
streamlit config show   Show settings
streamlit cache clear   Clear cache
streamlit --version     Show version
```

---

**🎉 Ready to use! Just run: `streamlit run app.py`**
