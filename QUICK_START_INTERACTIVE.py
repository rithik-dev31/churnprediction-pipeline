#!/usr/bin/env python3
"""
Quick Reference Script - How to Run Everything
Demonstrates all the ways to run the ML pipeline
"""

import pandas as pd
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def option_1_complete_pipeline():
    """Option 1: Run Complete Pipeline Script"""
    print_header("OPTION 1: RUN COMPLETE PIPELINE")
    print("""
    This runs the entire ML workflow end-to-end with detailed output.
    
    COMMAND:
    --------
    python run_full_pipeline.py
    
    WHAT IT DOES:
    - Loads data from datasets/churn_data.csv
    - Performs EDA (Exploratory Data Analysis)
    - Preprocessing & Feature Engineering
    - Splits data into train/test (80/20)
    - Trains 6 ML models
    - Evaluates all models
    - Saves best model
    
    OUTPUT:
    - logs/pipeline.log - Detailed logs
    - models/best_model.pkl - Best trained model
    - Console output with performance metrics
    
    ⏱️  TIME: ~2-5 minutes depending on data size
    """)


def option_2_streamlit_web():
    """Option 2: Streamlit Web Interface"""
    print_header("OPTION 2: STREAMLIT WEB INTERFACE (RECOMMENDED)")
    print("""
    Interactive web interface for the entire pipeline.
    
    SETUP:
    ------
    1. Install dependencies:
       pip install -r requirements.txt
    
    2. Run Streamlit:
       streamlit run app.py
    
    3. Browser opens at: http://localhost:8501
    
    FEATURES:
    - 🔐 Authentication (Sign up / Login)
    - 📤 Upload your own CSV data
    - 📊 Interactive EDA visualizations
    - 🤖 Train multiple models
    - 🎯 Real-time predictions
    - 📈 Model performance dashboard
    - 💾 Save/load trained models
    
    WORKFLOW:
    1. Sign up with username/password
    2. Go to "Upload Data" → Upload CSV file
    3. Go to "Explore Data" → View EDA charts
    4. Go to "Train Models" → Select dataset & train
    5. Go to "Make Predictions" → Test new data
    6. Go to "Model History" → View past training runs
    
    ⏱️  TIME: Interactive, run as needed
    """)


def option_3_python_script():
    """Option 3: Python Script (Programmatic)"""
    print_header("OPTION 3: PYTHON SCRIPT (PROGRAMMATIC)")
    print("""
    Control pipeline step-by-step from Python code.
    
    EXAMPLE CODE:
    -----
    
    from preprocessing import DataPreprocessor, train_test_split_stratified
    from train import ModelTrainer
    from evaluate import ModelEvaluator
    import pandas as pd
    
    # 1. Load data
    df = pd.read_csv('datasets/churn_data.csv')
    
    # 2. Preprocess with feature engineering
    prep = DataPreprocessor(target_col='Exited')
    X, y = prep.preprocess(df, create_features=True)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    
    # 4. Train models
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_all_models(X_train, y_train)
    
    # 5. Evaluate
    evaluator = ModelEvaluator()
    y_pred = trainer.predict('XGBoost', X_test)
    metrics = evaluator.evaluate(y_test, y_pred, 'XGBoost')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # 6. Make predictions on new data
    predictions = trainer.predict('XGBoost', X_test)
    probabilities = trainer.predict_proba('XGBoost', X_test)
    
    # 7. Save for later
    trainer.save_model('XGBoost', 'models/my_model.pkl')
    prep.save('models/preprocessor.pkl')
    -----
    
    ⏱️  TIME: Use as needed for custom workflows
    """)


def option_4_individual_scripts():
    """Option 4: Individual Scripts"""
    print_header("OPTION 4: INDIVIDUAL SCRIPTS")
    print("""
    Run specific scripts for specific tasks.
    
    COMMANDS:
    --------
    
    # Generate sample data for testing
    python generate_sample_data.py
    
    # Train models only
    python train.py
    
    # Make predictions on test data
    python predict.py
    
    # Evaluate trained models
    python evaluate.py
    
    ⏱️  TIME: Depends on script, typically 30 seconds - 2 minutes
    """)


def option_5_feature_engineering():
    """Option 5: Feature Engineering Examples"""
    print_header("OPTION 5: FEATURE ENGINEERING")
    print("""
    Different feature engineering techniques.
    
    ENABLED BY DEFAULT IN PREPROCESS:
    
    1. AGE_GROUP: Bins ages into Young, Mid Adult, Senior, Elderly
    2. CREDIT_GROUP: Categorizes credit scores (300-600, 600-700, etc.)
    3. HAS_BALANCE: Binary indicator of account balance
    4. TENURE_GROUP: Groups tenure into New, Regular, Loyal
    
    TO CUSTOMIZE:
    
    Edit preprocessing.py → create_engineered_features() method
    
    EXAMPLES:
    
    # Example 1: Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,40,50,100],
                             labels=['Young','Mid','Senior','Elderly'])
    
    # Example 2: Quantile-based binning
    df['Income_Quantile'] = pd.qcut(df['EstimatedSalary'], q=4)
    
    # Example 3: Ratio features
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    # Example 4: Custom function
    def categorize_balance(balance):
        if balance == 0: return 'None'
        elif balance < 50000: return 'Low'
        else: return 'High'
    df['Balance_Category'] = df['Balance'].apply(categorize_balance)
    
    TO DISABLE FEATURE ENGINEERING:
    
    # In preprocessing call, set create_features=False:
    X, y = prep.preprocess(df, create_features=False)
    """)


def quick_start_guide():
    """Print quick start guide."""
    print_header("⚡ QUICK START GUIDE")
    print("""
    Fastest way to get started in 3 steps:
    
    STEP 1: Install Dependencies
    ----------------------------
    pip install -r requirements.txt
    
    STEP 2: Choose Your Method
    --------------------------
    Option A - Complete Pipeline (All-in-one):
        python run_full_pipeline.py
    
    Option B - Web Interface (Interactive):
        streamlit run app.py
    
    STEP 3: View Results
    -------------------
    - Pipeline: Check console output + logs/pipeline.log
    - Streamlit: Open browser at http://localhost:8501
    
    ✅ DONE! Your models are trained and ready to use.
    """)


def troubleshooting():
    """Print troubleshooting tips."""
    print_header("🔧 TROUBLESHOOTING")
    print("""
    PROBLEM: "Module not found" error
    SOLUTION: Install requirements
        pip install -r requirements.txt
    
    PROBLEM: "Column not found" error
    SOLUTION: Update target column name in scripts
        - Change 'Exited' to 'Churn' or your column name
        - In run_full_pipeline.py, line ~224
    
    PROBLEM: Slow performance
    SOLUTION: 
        - Use smaller dataset first
        - Set feature_selection_k=10 to reduce features
        - Reduce test_size to smaller percentage
    
    PROBLEM: Memory error on large datasets
    SOLUTION:
        - Process data in batches
        - Use feature selection
        - Reduce n_estimators in RandomForest
    
    PROBLEM: Model performance is poor
    SOLUTION:
        - Try feature engineering
        - Tune hyperparameters
        - Use different models
        - Increase training data
    
    For more help, check:
    - FEATURE_ENGINEERING_GUIDE.md
    - logs/pipeline.log
    - README.md
    """)


def file_structure():
    """Show key files in project."""
    print_header("📁 KEY PROJECT FILES")
    print("""
    Your Pipeline Project Structure:
    
    pipeline/
    ├── app.py                         Main Streamlit application
    ├── run_full_pipeline.py           ⭐ Complete pipeline script
    ├── preprocessing.py               Data preprocessing & feature engineering
    ├── train.py                       Model training classes
    ├── evaluate.py                    Model evaluation metrics
    ├── predict.py                     Make predictions
    ├── database.py                    User & data management
    ├── config.py                      Configuration settings
    ├── requirements.txt               Python dependencies
    ├── FEATURE_ENGINEERING_GUIDE.md   ⭐ Detailed guide
    │
    ├── datasets/                      📂 Your data goes here
    │   └── churn_data.csv            (Place your CSV file here)
    │
    ├── models/                        📂 Trained models saved here
    │   ├── preprocessor.pkl
    │   └── xgboost_best.pkl
    │
    └── logs/                          📂 Application logs
        └── pipeline.log
    
    ⭐ = Most important for you to know about
    """)


def main():
    """Main menu."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CUSTOMER CHURN PREDICTION PIPELINE" + " " * 23 + "║")
    print("║" + " " * 25 + "Quick Reference & Setup Guide" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print("""
    Choose what you'd like to do:
    
    1. Run complete pipeline end-to-end
    2. Set up interactive web interface
    3. Use programmatic Python approach
    4. Run individual scripts
    5. Learn about feature engineering
    6. Quick start (3 steps)
    7. Troubleshooting tips
    8. View project file structure
    0. Exit
    """)
    
    choice = input("Enter your choice (0-8): ").strip()
    
    if choice == "1":
        option_1_complete_pipeline()
    elif choice == "2":
        option_2_streamlit_web()
    elif choice == "3":
        option_3_python_script()
    elif choice == "4":
        option_4_individual_scripts()
    elif choice == "5":
        option_5_feature_engineering()
    elif choice == "6":
        quick_start_guide()
    elif choice == "7":
        troubleshooting()
    elif choice == "8":
        file_structure()
    elif choice == "0":
        print("\nGoodbye! 👋\n")
        return
    else:
        print("\n❌ Invalid choice. Please try again.\n")
        main()


if __name__ == "__main__":
    main()
    
    # Ask if user wants to see more
    cont = input("\nWould you like to see other options? (y/n): ").strip().lower()
    if cont == 'y':
        main()
