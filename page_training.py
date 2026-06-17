"""
Model training page.
Feature engineering is applied to the raw dataset before feature selection and training.
Includes class imbalance handling: SMOTE, Oversample, Undersample, class_weight.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from src_auth import get_current_user_id, require_authentication
from database import get_user_datasets, save_model_info
from utils import load_csv, save_model_file
from preprocessing import DataPreprocessor, train_test_split_stratified
from train import ModelTrainer, ModelHyperparameterTuner
from evaluate import ModelEvaluator
from feature_engineering import apply_feature_engineering, get_feature_groups
import plotly.graph_objects as go


# ── Imbalance handling helpers ─────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """Oversample minority class with SMOTE on training data only."""
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def apply_oversample(X_train, y_train):
    """Random oversample minority class to match majority."""
    df = pd.concat([X_train, y_train], axis=1)
    target = y_train.name
    majority = df[df[target] == 0]
    minority = df[df[target] == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df_balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
    return df_balanced.drop(columns=[target]), df_balanced[target]


def apply_undersample(X_train, y_train):
    """Random undersample majority class to match minority."""
    df = pd.concat([X_train, y_train], axis=1)
    target = y_train.name
    majority = df[df[target] == 0]
    minority = df[df[target] == 1]
    majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    df_balanced = pd.concat([majority_down, minority]).sample(frac=1, random_state=42)
    return df_balanced.drop(columns=[target]), df_balanced[target]


def get_class_weight_models():
    """Return model names that support class_weight='balanced'."""
    return {'Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM'}


def reinitialize_models_with_class_weight(trainer):
    """Reinitialize trainer models with class_weight='balanced'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    trainer.models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            n_jobs=-1, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=42, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42, verbosity=0,
            # XGBoost uses scale_pos_weight instead of class_weight
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42, class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),  # no class_weight support
    }


# ── Main page ──────────────────────────────────────────────────────────────────

def show_training_page():
    """Display model training page."""
    st.title("🤖 Train Models")
    st.markdown("Train and evaluate multiple ML models")

    require_authentication()
    user_id = get_current_user_id()

    datasets = get_user_datasets(user_id)
    if not datasets:
        st.warning("No datasets found. Please upload a dataset first.")
        return

    dataset_names = {f"{d['dataset_name']} ({d['rows']} rows)": d['file_path']
                     for d in datasets}
    selected_dataset_label = st.selectbox("Select Dataset", list(dataset_names.keys()))

    dataset_path = dataset_names[selected_dataset_label]
    raw_df = load_csv(dataset_path)

    if raw_df is None:
        st.error("Failed to load dataset")
        return

    # ── Feature engineering ────────────────────────────────────────────────────
    df = apply_feature_engineering(raw_df)
    feature_groups = get_feature_groups()

    target_col = feature_groups['target']
    if target_col not in df.columns:
        if 'Churn' in df.columns:
            target_col = 'Churn'
        else:
            st.error("Dataset must contain 'Exited' or 'Churn' column as target variable")
            return

    # ── Class imbalance warning ────────────────────────────────────────────────
    class_counts = df[target_col].value_counts()
    minority_pct = class_counts.min() / len(df) * 100
    majority_pct = class_counts.max() / len(df) * 100
    imbalance_ratio = class_counts.max() / class_counts.min()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("✅ Not Churned (0)", f"{class_counts.get(0, 0):,} ({majority_pct:.1f}%)")
    with col_b:
        st.metric("🔴 Churned (1)", f"{class_counts.get(1, 0):,} ({minority_pct:.1f}%)")
    with col_c:
        st.metric("⚖️ Imbalance Ratio", f"{imbalance_ratio:.1f}x")

    if imbalance_ratio > 1.5:
        st.warning(
            f"⚠️ Dataset is imbalanced ({imbalance_ratio:.1f}x ratio). "
            "This causes low Recall & Precision on the minority class. "
            "Use a balancing strategy below."
        )

    st.info(
        f"✅ Feature engineering applied — "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns "
        f"(includes: {', '.join(feature_groups['engineered'])})"
    )

    st.divider()

    # ── Feature selection ──────────────────────────────────────────────────────
    st.subheader("📊 Feature Selection")

    available_features = [col for col in df.columns if col != target_col]
    default_features   = [f for f in feature_groups['default_training'] if f in available_features]

    selected_features = st.multiselect(
        "Select Features for Training",
        available_features,
        default=default_features,
        help="Engineered features (Age_Group, credit_label, etc.) are pre-selected.",
    )

    if not selected_features:
        st.warning("Please select at least one feature for training")
        return

    st.write(f"✓ Selected {len(selected_features)} features: {', '.join(selected_features)}")

    st.divider()

    # ── Preprocessing configuration ────────────────────────────────────────────
    st.subheader("⚙️ Preprocessing Configuration")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        missing_method = st.selectbox("Missing Values", ["mean", "median", "drop"])
    with col2:
        encoding_method = st.selectbox("Encoding", ["label", "onehot"])
    with col3:
        scaling_method = st.selectbox("Scaling", ["standard", "minmax"])
    with col4:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    st.divider()

    # ── Class imbalance strategy ───────────────────────────────────────────────
    st.subheader("⚖️ Class Imbalance Strategy")

    imbalance_strategy = st.radio(
        "Select strategy to handle class imbalance",
        [
            "None (no balancing)",
            "SMOTE — synthetic oversampling of minority class  ✅ Recommended",
            "Oversample — random duplicate minority samples",
            "Undersample — randomly reduce majority class",
            "Class Weight — penalize misclassifying minority class",
        ],
        index=1,  # SMOTE pre-selected
        help=(
            "SMOTE: best balance between recall and precision.\n"
            "Oversample: simple, no data loss.\n"
            "Undersample: faster training but loses majority data.\n"
            "Class Weight: no resampling, adjusts model loss function."
        ),
    )

    # Parse strategy key
    if "SMOTE" in imbalance_strategy:
        strategy = "smote"
    elif "Oversample" in imbalance_strategy:
        strategy = "oversample"
    elif "Undersample" in imbalance_strategy:
        strategy = "undersample"
    elif "Class Weight" in imbalance_strategy:
        strategy = "class_weight"
    else:
        strategy = "none"

    # Show strategy explanation
    strategy_info = {
        "smote":        "🧬 SMOTE generates synthetic minority samples using k-nearest neighbours. Applied to training set only.",
        "oversample":   "📋 Randomly duplicates existing minority samples until classes are equal. Applied to training set only.",
        "undersample":  "✂️ Randomly removes majority samples until classes are equal. May lose useful data.",
        "class_weight": "⚖️ Adjusts the model's loss function to penalise minority misclassification more heavily. No resampling.",
        "none":         "⚠️ No balancing applied. Models may be biased toward the majority class (low Recall on churners).",
    }
    st.info(strategy_info[strategy])

    st.divider()

    # ── Model selection ────────────────────────────────────────────────────────
    st.subheader("🎯 Select Models to Train")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_lr  = st.checkbox("Logistic Regression", value=True)
        train_rf  = st.checkbox("Random Forest",        value=True)
    with col2:
        train_dt  = st.checkbox("Decision Tree",        value=True)
        train_xgb = st.checkbox("XGBoost",              value=True)
    with col3:
        train_svm = st.checkbox("SVM",                  value=True)
        train_knn = st.checkbox("KNN",                  value=True)

    models_to_train = []
    if train_lr:  models_to_train.append("Logistic Regression")
    if train_rf:  models_to_train.append("Random Forest")
    if train_dt:  models_to_train.append("Decision Tree")
    if train_xgb: models_to_train.append("XGBoost")
    if train_svm: models_to_train.append("SVM")
    if train_knn: models_to_train.append("KNN")

    # ── Advanced options ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Advanced Options")

    col1, col2 = st.columns(2)
    with col1:
        enable_feature_selection = st.checkbox("Enable Feature Selection", value=False)
        feature_selection_k = None
        if enable_feature_selection:
            feature_selection_k = st.slider("Number of Features to Select", 5, 50, 10)
    with col2:
        enable_cross_validation = st.checkbox("Enable Cross-Validation", value=True)
        cv_folds = 5
        if enable_cross_validation:
            cv_folds = st.slider("CV Folds", 3, 10, 5)

    st.divider()

    if st.button("🚀 Start Training", use_container_width=True, key="train_btn"):
        train_models(
            df, user_id, dataset_path,
            missing_method, encoding_method, scaling_method, test_size,
            models_to_train, enable_feature_selection, feature_selection_k,
            enable_cross_validation, cv_folds, target_col, selected_features,
            strategy,
        )


# ── Training logic ─────────────────────────────────────────────────────────────

def train_models(df, user_id, dataset_path, missing_method, encoding_method,
                 scaling_method, test_size, models_to_train, enable_feature_selection,
                 feature_selection_k, enable_cross_validation, cv_folds,
                 target_col, selected_features, imbalance_strategy):
    """Train selected models with imbalance handling."""

    progress_bar      = st.progress(0)
    status_text       = st.empty()
    results_container = st.container()

    try:
        # Step 1: Subset
        status_text.text("📊 Selecting features...")
        progress_bar.progress(5)
        df_selected = df[selected_features + [target_col]].copy()

        # Step 2: Preprocess
        status_text.text("🔄 Preprocessing data...")
        progress_bar.progress(15)

        preprocessor = DataPreprocessor(target_col=target_col)
        X, y = preprocessor.preprocess(
            df_selected, is_training=True,
            missing_method=missing_method,
            encoding_method=encoding_method,
            scaling_method=scaling_method,
            feature_selection_k=feature_selection_k if enable_feature_selection else None,
            create_features=False,
        )

        # Step 3: Train/test split (split BEFORE balancing to avoid data leakage)
        status_text.text("✂️ Splitting data...")
        progress_bar.progress(25)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=test_size
        )

        # Step 4: Apply imbalance strategy to TRAINING SET ONLY
        original_train_dist = y_train.value_counts().to_dict()

        if imbalance_strategy == "smote":
            status_text.text("🧬 Applying SMOTE...")
            X_train, y_train = apply_smote(X_train, y_train)

        elif imbalance_strategy == "oversample":
            status_text.text("📋 Oversampling minority class...")
            X_train, y_train = apply_oversample(X_train, y_train)

        elif imbalance_strategy == "undersample":
            status_text.text("✂️ Undersampling majority class...")
            X_train, y_train = apply_undersample(X_train, y_train)

        new_train_dist = y_train.value_counts().to_dict()
        st.info(
            f"📊 Training set — "
            f"Before: {original_train_dist} → "
            f"After: {new_train_dist}"
        )

        # Step 5: Initialize models (with class_weight if selected)
        trainer = ModelTrainer()
        if imbalance_strategy == "class_weight":
            reinitialize_models_with_class_weight(trainer)
        else:
            trainer.initialize_models()

        evaluator = ModelEvaluator()
        all_metrics = {}

        for i, model_name in enumerate(models_to_train):
            progress = 25 + (i + 1) * (60 // len(models_to_train))
            status_text.text(f"🤖 Training {model_name}...")
            progress_bar.progress(min(progress, 95))

            success, message = trainer.train_model(model_name, X_train, y_train)

            if success:
                y_pred  = trainer.predict(model_name, X_test)
                metrics = evaluator.calculate_metrics(y_test, y_pred, model_name)
                all_metrics[model_name] = metrics

                if enable_cross_validation:
                    cv_scores = trainer.cross_validate(
                        model_name, X_train, y_train, cv=cv_folds
                    )
                    metrics['cv_mean'] = cv_scores['mean_score']
                    metrics['cv_std']  = cv_scores['std_score']

                model_path = f"models/{user_id}_{model_name.replace(' ', '_')}.joblib"

                trained_package = {
                    "model": trainer.get_model(model_name),
                    "preprocessor": preprocessor,
                    "feature_columns": X.columns.tolist(),
                    "selected_features": selected_features,
                    "target_column": target_col
                }

                save_model_file(trained_package, model_path)
                save_model_info(
                    user_id, model_name, "classifier", model_path,
                    metrics['accuracy'], metrics['precision'],
                    metrics['recall'],   metrics['f1_score'],
                )
            else:
                st.warning(f"⚠️ {model_name} failed: {message}")

        if not all_metrics:
            st.error("No models trained successfully. Check the warnings above.")
            progress_bar.progress(0)
            status_text.text("")
            return

        progress_bar.progress(100)
        status_text.text("✅ Training completed!")

        # ── Results ────────────────────────────────────────────────────────────
        with results_container:
            st.success("Training completed successfully!")
            st.divider()

            st.subheader("📊 Model Comparison")

            metrics_df = pd.DataFrame(all_metrics).T

            # Colour-code recall/precision columns to highlight improvement
            def highlight_recall(val):
                if isinstance(val, float):
                    if val >= 0.7:
                        return 'background-color: #d4edda; color: #155724'
                    elif val >= 0.5:
                        return 'background-color: #fff3cd; color: #856404'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                return ''

            styled = metrics_df.style.applymap(
                highlight_recall,
                subset=[c for c in ['recall', 'precision', 'f1_score'] if c in metrics_df.columns]
            ).format("{:.4f}", subset=metrics_df.select_dtypes('number').columns)

            st.dataframe(styled, use_container_width=True)

            # Best model by F1
            best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("🏆 Best Model", best_model[0])
            with col2: st.metric("Accuracy",  f"{best_model[1]['accuracy']:.2%}")
            with col3: st.metric("Precision", f"{best_model[1]['precision']:.2%}")
            with col4: st.metric("Recall",    f"{best_model[1]['recall']:.2%}")
            with col5: st.metric("F1-Score",  f"{best_model[1]['f1_score']:.2%}")

            # Bar chart
            fig = go.Figure()
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                fig.add_trace(go.Bar(
                    name=metric_name.replace('_', ' ').title(),
                    x=list(all_metrics.keys()),
                    y=[all_metrics[m].get(metric_name, 0) for m in all_metrics],
                ))
            fig.update_layout(
                title=f"Model Performance Comparison — Strategy: {imbalance_strategy.upper()}",
                barmode='group',
                xaxis_title="Model",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # CV scores if enabled
            if enable_cross_validation:
                cv_data = {
                    m: {'CV Mean': v.get('cv_mean', 0), 'CV Std': v.get('cv_std', 0)}
                    for m, v in all_metrics.items()
                    if 'cv_mean' in v
                }
                if cv_data:
                    st.subheader("🔄 Cross-Validation Scores")
                    st.dataframe(pd.DataFrame(cv_data).T, use_container_width=True)

    except Exception as e:
        import traceback
        st.error(f"❌ Error during training: {str(e)}")
        st.code(traceback.format_exc())
        progress_bar.progress(0)
        status_text.text("")