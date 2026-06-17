"""
Feature Engineering module.
Runs BEFORE EDA and training — produces the canonical enriched DataFrame
with all engineered columns from the raw Churn_Modelling dataset.

Expected input columns (raw CSV after dropping RowNumber/CustomerId/Surname):
    CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited

Output adds 5 engineered columns:
    Age_Group, Credit_Group, balance_label, credit_label, salary_label

Final column order:
    CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
    Exited, Age_Group, Credit_Group, balance_label, credit_label, salary_label
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Columns to drop from raw CSV ──────────────────────────────────────────────
_DROP_COLS = ['RowNumber', 'CustomerId', 'Surname']

# ── Canonical feature order (matches what training page expects) ───────────────
ENGINEERED_COLUMNS = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Exited',
    'Age_Group', 'Credit_Group', 'balance_label', 'credit_label', 'salary_label',
]

# ── Feature definitions ────────────────────────────────────────────────────────

def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age_Group: bucket Age into 4 life-stage segments.
        Young       → 0–30
        Mid Adult   → 31–40
        Senior      → 41–50
        Elderly     → 51+
    """
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 40, 50, 100],
        labels=['Young', 'Mid Adult', 'Senior', 'Elderly'],
        include_lowest=True,
    )
    return df


def _add_credit_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit_Group: fine-grained CreditScore tiers (industry standard).
        Very Poor   → 300–579
        Fair        → 580–669
        Good        → 670–739
        Very Good   → 740–799
        Exceptional → 800–849
        Elite       → 850+
    """
    df['Credit_Group'] = pd.cut(
        df['CreditScore'],
        bins=[0, 300, 580, 670, 740, 800, 850],
        labels=['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional', 'Elite'],
        include_lowest=True,
    )
    return df


def _add_balance_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    balance_label: account Balance tier.
        Zero      → 0
        Low       → 0–50k
        Medium    → 50k–100k
        High      → 100k–150k
        Very High → 150k+
    """
    df['balance_label'] = pd.cut(
        df['Balance'],
        bins=[-1, 0, 50_000, 100_000, 150_000, 300_000],
        labels=['Zero', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True,
    )
    return df


def _add_credit_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    credit_label: simplified 4-tier CreditScore label (used as a training feature).
        Poor      → ≤600
        Fair      → 601–700
        Good      → 701–800
        Excellent → 801+
    """
    df['credit_label'] = pd.cut(
        df['CreditScore'],
        bins=[0, 600, 700, 800, 900],
        labels=['Poor', 'Fair', 'Good', 'Excellent'],
        include_lowest=True,
    )
    return df


def _add_salary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    salary_label: EstimatedSalary tier.
        Low       → 0–50k
        Medium    → 50k–100k
        High      → 100k–150k
        Very High → 150k+
    """
    df['salary_label'] = pd.cut(
        df['EstimatedSalary'],
        bins=[0, 50_000, 100_000, 150_000, 200_000],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True,
    )
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the raw DataFrame.

    - Drops identifier columns (RowNumber, CustomerId, Surname) if present.
    - Adds: Age_Group, Credit_Group, balance_label, credit_label, salary_label.
    - Returns columns in canonical ENGINEERED_COLUMNS order
      (extra columns not in the list are kept at the end).

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        Enriched DataFrame ready for EDA and model training.
    """
    df_out = df.copy()

    # Drop identifier columns if still present
    drop_existing = [c for c in _DROP_COLS if c in df_out.columns]
    if drop_existing:
        df_out = df_out.drop(columns=drop_existing)
        logger.info(f"Dropped identifier columns: {drop_existing}")

    # Apply each engineering step
    df_out = _add_age_group(df_out)
    df_out = _add_credit_group(df_out)
    df_out = _add_balance_label(df_out)
    df_out = _add_credit_label(df_out)
    df_out = _add_salary_label(df_out)

    # Reorder to canonical column order (keep any extra cols at the end)
    ordered = [c for c in ENGINEERED_COLUMNS if c in df_out.columns]
    extra   = [c for c in df_out.columns if c not in ordered]
    df_out  = df_out[ordered + extra]

    logger.info(
        f"Feature engineering complete. "
        f"Shape: {df_out.shape} | "
        f"New columns: Age_Group, Credit_Group, balance_label, credit_label, salary_label"
    )
    return df_out


def get_feature_groups() -> dict:
    """
    Return feature groups for use in EDA and training pages.

    Returns:
        dict with keys: 'numerical', 'categorical', 'engineered', 'target', 'default_training'
    """
    return {
        'numerical': [
            'CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        ],
        'categorical': [
            'Geography', 'Gender',
        ],
        'engineered': [
            'Age_Group', 'Credit_Group', 'balance_label', 'credit_label', 'salary_label',
        ],
        'target': 'Exited',
        # Recommended features pre-selected on the training page
        'default_training': [
            'CreditScore', 'Gender', 'Age', 'Balance', 'EstimatedSalary',
            'credit_label', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Age_Group',
        ],
    }