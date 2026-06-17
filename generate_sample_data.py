"""
Sample dataset generator for testing the application.
Generates synthetic customer churn data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_dataset(n_samples=1000, random_state=42):
    """
    Generate a sample customer churn dataset.
    
    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with customer data and churn indicator
    """
    np.random.seed(random_state)
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 75, n_samples),
        'Tenure': np.random.randint(0, 72, n_samples),  # months
        'MonthlyCharges': np.random.uniform(20, 150, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Internet Service': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Credit card', 'Bank transfer', 'Mailed check', 'Electronic check'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate Churn based on features (realistic pattern)
    churn_probability = (
        (df['Tenure'] < 12).astype(int) * 0.3 +  # New customers more likely to churn
        (df['MonthlyCharges'] > 100).astype(int) * 0.2 +  # High charges increase churn risk
        (df['Contract'] == 'Month-to-month').astype(int) * 0.25 +  # Flexible contracts more likely
        (df['OnlineSecurity'] == 'No').astype(int) * 0.15  # Security adoption reduces churn
    )
    
    # Add randomness
    churn_probability += np.random.uniform(0, 0.1, n_samples)
    churn_probability = np.clip(churn_probability, 0, 1)
    
    df['Churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df


def save_sample_dataset(filepath='sample_data.csv', n_samples=1000):
    """
    Generate and save sample dataset.
    
    Args:
        filepath: Where to save the CSV
        n_samples: Number of records
    """
    df = generate_sample_dataset(n_samples)
    df.to_csv(filepath, index=False)
    print(f"✅ Sample dataset saved to: {filepath}")
    print(f"   - Samples: {len(df)}")
    print(f"   - Features: {df.shape[1] - 1}")
    print(f"   - Churn Rate: {df['Churn'].mean():.2%}")
    return df


def get_dataset_info(df):
    """Print dataset information."""
    print("\n📊 Dataset Information:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"   Missing: {df.isnull().sum().sum()} values")
    print(f"   Duplicates: {df.duplicated().sum()} rows")
    print(f"\n📈 Target Variable Distribution:")
    print(df['Churn'].value_counts())
    print(f"\n   No Churn: {(df['Churn']==0).sum()} ({(df['Churn']==0).mean():.2%})")
    print(f"   Churn: {(df['Churn']==1).sum()} ({(df['Churn']==1).mean():.2%})")


if __name__ == "__main__":
    # Generate and save sample dataset
    print("🚀 Generating sample customer churn dataset...")
    df = save_sample_dataset('sample_data.csv', n_samples=1000)
    
    # Print information
    get_dataset_info(df)
    
    # Show first few rows
    print("\n📋 First 5 rows:")
    print(df.head())
    
    print("\n✅ Sample dataset ready for testing!")
    print("   Upload 'sample_data.csv' in the application to get started.")
