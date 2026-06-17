"""
Complete ML Pipeline: Load → Clean → EDA → Preprocess → Train → Evaluate

This script runs the entire machine learning pipeline from data loading to model evaluation.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import DataPreprocessor, train_test_split_stratified
from train import ModelTrainer, ModelHyperparameterTuner
from evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def step_1_load_data(file_path):
    """Load dataset from CSV."""
    print("\n" + "=" * 80)
    print("[STEP 1/7] LOADING DATA")
    print("=" * 80)
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"  File: {file_path}")
        print(f"  Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def step_2_eda(df):
    """Exploratory Data Analysis."""
    print("\n" + "=" * 80)
    print("[STEP 2/7] EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n📋 Data Types:")
    print(df.dtypes)
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  None - Dataset is complete!")
    else:
        print(missing[missing > 0])
    
    print(f"\n📈 Numerical Features Statistics:")
    print(df.describe().T)
    
    print(f"\n🔤 Categorical Features:")
    for col in df.select_dtypes(include='object').columns:
        print(f"  {col}: {df[col].nunique()} unique values - {df[col].value_counts().to_dict()}")
    
    return True


def step_3_preprocessing(df, target_col='Exited', create_features=True):
    """Data Preprocessing with Feature Engineering."""
    print("\n" + "=" * 80)
    print("[STEP 3/7] DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 80)
    
    try:
        preprocessor = DataPreprocessor(target_col=target_col)
        
        print(f"\n  Processing steps:")
        print(f"    ✓ Missing value handling: mean")
        print(f"    ✓ Feature engineering: {create_features}")
        print(f"    ✓ Categorical encoding: label")
        print(f"    ✓ Feature scaling: standard")
        
        X_processed, y = preprocessor.preprocess(
            df,
            is_training=True,
            missing_method='mean',
            encoding_method='label',
            scaling_method='standard',
            feature_selection_k=None,
            create_features=create_features
        )
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Output shape: {X_processed.shape[0]} rows × {X_processed.shape[1]} features")
        print(f"  Features: {', '.join(X_processed.columns[:5])}{'...' if len(X_processed.columns) > 5 else ''}")
        
        if y is not None:
            print(f"\n  Target variable distribution:")
            for label, count in y.value_counts().items():
                pct = (count / len(y)) * 100
                print(f"    {label}: {count:,} ({pct:.1f}%)")
        
        return X_processed, y, preprocessor
    
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def step_4_split_data(X, y, test_size=0.2):
    """Train-Test Split."""
    print("\n" + "=" * 80)
    print("[STEP 4/7] TRAIN-TEST SPLIT")
    print("=" * 80)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y,
            test_size=test_size,
            random_state=42
        )
        
        print(f"\n✓ Data split complete!")
        print(f"  Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Test set:     {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
        print(f"  Features:     {X_train.shape[1]}")
        
        print(f"\n  Train target distribution:")
        for label, count in y_train.value_counts().items():
            pct = (count / len(y_train)) * 100
            print(f"    {label}: {count:,} ({pct:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"✗ Error in splitting: {e}")
        return None, None, None, None


def step_5_train_models(X_train, y_train):
    """Train all models."""
    print("\n" + "=" * 80)
    print("[STEP 5/7] MODEL TRAINING")
    print("=" * 80)
    
    try:
        trainer = ModelTrainer()
        trainer.initialize_models()
        
        print(f"\n  Initializing {len(trainer.models)} models:")
        for model_name in trainer.models.keys():
            print(f"    • {model_name}")
        
        print(f"\n  Training models...")
        results = trainer.train_all_models(X_train, y_train)
        
        print(f"\n✓ Training complete!")
        successful = sum(1 for success, _ in results.values() if success)
        
        for model_name, (success, message) in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {model_name:20s} {message}")
        
        print(f"\n  Summary: {successful}/{len(results)} models trained successfully")
        
        return trainer
    
    except Exception as e:
        print(f"✗ Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_6_evaluate_models(trainer, X_test, y_test):
    """Evaluate all trained models."""
    print("\n" + "=" * 80)
    print("[STEP 6/7] MODEL EVALUATION")
    print("=" * 80)
    
    try:
        evaluator = ModelEvaluator()
        evaluation_results = {}
        
        print(f"\n  Evaluating trained models:\n")
        
        for model_name in trainer.trained_models.keys():
            try:
                y_pred = trainer.predict(model_name, X_test)
                metrics = evaluator.evaluate(y_test, y_pred, model_name)
                evaluation_results[model_name] = metrics
                
                print(f"  📊 {model_name}")
                print(f"      Accuracy:  {metrics['accuracy']:.4f}")
                print(f"      Precision: {metrics['precision']:.4f}")
                print(f"      Recall:    {metrics['recall']:.4f}")
                print(f"      F1 Score:  {metrics['f1']:.4f}")
                print(f"      AUC-ROC:   {metrics['roc_auc']:.4f}")
                print()
            
            except Exception as e:
                print(f"  ✗ Error evaluating {model_name}: {e}\n")
        
        if evaluation_results:
            print(f"✓ Evaluation complete!")
            return evaluation_results
        else:
            print(f"✗ No models could be evaluated")
            return None
    
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_7_best_model(trainer, evaluation_results):
    """Select and save best model."""
    print("\n" + "=" * 80)
    print("[STEP 7/7] BEST MODEL SELECTION & SAVING")
    print("=" * 80)
    
    try:
        if not evaluation_results:
            print("✗ No evaluation results available")
            return None
        
        # Find best model by F1 Score
        best_model_name = max(
            evaluation_results.items(),
            key=lambda x: x[1]['f1']
        )[0]
        
        best_metrics = evaluation_results[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   F1 Score:  {best_metrics['f1']:.4f}")
        print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall:    {best_metrics['recall']:.4f}")
        print(f"   AUC-ROC:   {best_metrics['roc_auc']:.4f}")
        
        # Create models directory if it doesn't exist
        Path('models').mkdir(exist_ok=True)
        
        # Save best model
        model_path = f'models/{best_model_name.replace(" ", "_").lower()}_best.pkl'
        trainer.save_model(best_model_name, model_path)
        print(f"\n✓ Best model saved: {model_path}")
        
        return best_model_name
    
    except Exception as e:
        print(f"✗ Error selecting best model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run complete ML pipeline."""
    
    print("\n" + "=" * 80)
    print("🚀 CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    print("=" * 80)
    
    # Configuration
    DATA_FILE = 'datasets/churn_data.csv'  # Change to your file path
    TARGET_COL = 'Exited'  # or 'Churn' depending on your data
    CREATE_FEATURES = True  # Enable feature engineering
    TEST_SIZE = 0.2
    
    # Step 1: Load Data
    df = step_1_load_data(DATA_FILE)
    if df is None:
        print("\n❌ Pipeline failed: Could not load data")
        return
    
    # Step 2: EDA
    step_2_eda(df)
    
    # Step 3: Preprocessing & Feature Engineering
    X_processed, y, preprocessor = step_3_preprocessing(
        df,
        target_col=TARGET_COL,
        create_features=CREATE_FEATURES
    )
    
    if X_processed is None:
        print("\n❌ Pipeline failed: Preprocessing error")
        return
    
    # Step 4: Split Data
    X_train, X_test, y_train, y_test = step_4_split_data(
        X_processed, y,
        test_size=TEST_SIZE
    )
    
    if X_train is None:
        print("\n❌ Pipeline failed: Data split error")
        return
    
    # Step 5: Train Models
    trainer = step_5_train_models(X_train, y_train)
    if trainer is None or not trainer.trained_models:
        print("\n❌ Pipeline failed: Model training error")
        return
    
    # Step 6: Evaluate Models
    evaluation_results = step_6_evaluate_models(trainer, X_test, y_test)
    if evaluation_results is None:
        print("\n❌ Pipeline failed: Model evaluation error")
        return
    
    # Step 7: Best Model
    best_model = step_7_best_model(trainer, evaluation_results)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Outputs saved in:")
    print(f"  • Models: models/")
    print(f"  • Logs: logs/pipeline.log")
    print(f"\n📊 Best Model: {best_model}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review model performance in logs/pipeline.log")
    print(f"  2. Use trained model for predictions")
    print(f"  3. Optimize hyperparameters if needed")
    print(f"  4. Deploy model to production")
    print("\n" + "=" * 80 + "\n")
    
    return trainer, preprocessor, evaluation_results


if __name__ == "__main__":
    trainer, preprocessor, results = main()
