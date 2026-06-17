"""
EDA (Exploratory Data Analysis) page.
Feature engineering is applied to the raw dataset before any analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src_auth import get_current_user_id, require_authentication
from database import get_user_datasets
from utils import load_csv, get_numerical_columns, get_categorical_columns, detect_outliers
from feature_engineering import apply_feature_engineering, get_feature_groups


def show_eda_page():
    """Display EDA page."""
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.markdown("Analyze and visualize your data")

    require_authentication()
    user_id = get_current_user_id()

    # Get user datasets
    datasets = get_user_datasets(user_id)

    if not datasets:
        st.warning("No datasets found. Please upload a dataset first.")
        return

    # Select dataset
    dataset_names = {f"{d['dataset_name']} ({d['rows']} rows)": d['file_path']
                     for d in datasets}
    selected_dataset_label = st.selectbox("Select Dataset", list(dataset_names.keys()))

    dataset_path = dataset_names[selected_dataset_label]
    raw_df = load_csv(dataset_path)

    if raw_df is None:
        st.error("Failed to load dataset")
        return

    # ── Apply feature engineering BEFORE any EDA ──────────────────────────────
    df = apply_feature_engineering(raw_df)
    feature_groups = get_feature_groups()

    st.success(
        f"✅ Feature engineering applied — "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns "
        f"(+{len(feature_groups['engineered'])} engineered features)"
    )

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Distributions", "Correlations", "Relationships", "Outliers", "Target Analysis"
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col3:
            st.metric("Duplicates", df.duplicated().sum())

        st.write("**Data Types:**")
        st.dataframe(pd.DataFrame({
            'Column':    df.columns,
            'Type':      df.dtypes.astype(str),
            'Non-Null':  df.count(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        }), use_container_width=True)

        st.write("**Missing Values:**")
        missing_df = pd.DataFrame({
            'Column':     df.columns,
            'Count':      df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        }).sort_values('Count', ascending=False)

        if missing_df['Count'].sum() > 0:
            st.dataframe(missing_df[missing_df['Count'] > 0], use_container_width=True)
        else:
            st.success("No missing values detected!")

        # Show engineered feature summary
        st.write("**Engineered Features:**")
        eng_cols = feature_groups['engineered']
        eng_summary = []
        for col in eng_cols:
            if col in df.columns:
                vc = df[col].value_counts()
                eng_summary.append({
                    'Feature':    col,
                    'Categories': ', '.join(vc.index.astype(str).tolist()),
                    'Counts':     ', '.join(vc.values.astype(str).tolist()),
                })
        if eng_summary:
            st.dataframe(pd.DataFrame(eng_summary), use_container_width=True)

    # ── Tab 2: Distributions ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Distributions")

        num_cols = get_numerical_columns(df)

        if num_cols:
            selected_col = st.selectbox("Select numerical column", num_cols, key="dist_col")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_col, nbins=30,
                                   title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y=selected_col,
                             title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

            st.write("**Statistics:**")
            st.dataframe(df[selected_col].describe())

        # Categorical + engineered feature distributions
        # Include both raw categoricals AND engineered (category dtype) columns
        cat_cols = get_categorical_columns(df)
        eng_cols = [c for c in feature_groups['engineered'] if c in df.columns]
        all_cat = list(dict.fromkeys(cat_cols + eng_cols))  # deduplicated, ordered

        if all_cat:
            st.write("---")
            st.write("**Categorical & Engineered Features:**")

            selected_cat = st.selectbox("Select categorical column", all_cat, key="cat_col")

            cat_counts = df[selected_cat].value_counts().reset_index()
            cat_counts.columns = [selected_cat, 'count']

            fig = px.bar(
                cat_counts,
                x=selected_cat, y='count',
                title=f"Distribution of {selected_cat}",
                labels={'count': 'Count', selected_cat: selected_cat},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Churn rate by engineered feature
            target = feature_groups['target']
            if target in df.columns and selected_cat in eng_cols:
                st.write(f"**Churn Rate by {selected_cat}:**")
                churn_rate = (
                    df.groupby(selected_cat, observed=True)[target]
                    .mean()
                    .reset_index()
                    .rename(columns={target: 'Churn Rate'})
                )
                churn_rate['Churn Rate'] = (churn_rate['Churn Rate'] * 100).round(2)
                fig2 = px.bar(
                    churn_rate,
                    x=selected_cat, y='Churn Rate',
                    title=f"Churn Rate (%) by {selected_cat}",
                    labels={'Churn Rate': 'Churn Rate (%)'},
                    color='Churn Rate',
                    color_continuous_scale='RdYlGn_r',
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Correlations ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Correlation Analysis")

        num_cols = get_numerical_columns(df)

        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            plt.title("Correlation Heatmap")
            st.pyplot(fig)

            st.write("**High Correlations (> 0.7):**")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Feature 1':   corr_matrix.columns[i],
                            'Feature 2':   corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j],
                        })
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info("No correlations > 0.7 found")
        else:
            st.warning("Need at least 2 numerical columns for correlation analysis")

    # ── Tab 4: Relationships ───────────────────────────────────────────────────
    with tab4:
        st.subheader("Feature Relationships")

        num_cols = get_numerical_columns(df)

        if len(num_cols) >= 2:
            col1 = st.selectbox("X-axis", num_cols, key="scatter_x")
            col2 = st.selectbox("Y-axis", num_cols, key="scatter_y",
                                index=1 if len(num_cols) > 1 else 0)

            target = feature_groups['target']
            color_col = target if target in df.columns else None

            fig = px.scatter(
                df, x=col1, y=col2,
                color=color_col,
                title=f"{col1} vs {col2}",
                opacity=0.6,
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'} if color_col else None,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Outliers ────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Outlier Detection")

        num_cols = get_numerical_columns(df)

        if num_cols:
            method = st.radio("Detection Method", ["IQR", "Z-Score"])

            outlier_rows = []
            for col in num_cols:
                outliers = detect_outliers(
                    df[col], method='iqr' if method == "IQR" else 'zscore'
                )
                outlier_count = int(outliers.sum()) if isinstance(outliers, pd.Series) else int(outliers.shape[0])
                outlier_rows.append({
                    'Feature':   col,
                    'Outliers':  outlier_count,
                    'Percentage': f"{outlier_count / len(df) * 100:.2f}%",
                })

            outlier_df = pd.DataFrame(outlier_rows).sort_values('Outliers', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)
            st.info(f"Using {method} method for outlier detection")

    # ── Tab 6: Target Analysis ─────────────────────────────────────────────────
    with tab6:
        st.subheader("Target Variable Analysis")

        target = feature_groups['target']

        if target in df.columns:
            exited_dist = df[target].value_counts().sort_index()

            fig = go.Figure(data=[go.Pie(
                labels=['Not Exited (0)', 'Exited (1)'],
                values=exited_dist.values,
                hole=0.3,
                marker_colors=['#2ecc71', '#e74c3c'],
            )])
            fig.update_layout(title="Target Variable Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Churn rate by every engineered feature
            st.write("**Churn Rate by Engineered Features:**")
            eng_cols = [c for c in feature_groups['engineered'] if c in df.columns]

            for eng_col in eng_cols:
                churn_by = (
                    df.groupby(eng_col, observed=True)[target]
                    .agg(['mean', 'count'])
                    .reset_index()
                    .rename(columns={'mean': 'Churn Rate', 'count': 'Count'})
                )
                churn_by['Churn Rate %'] = (churn_by['Churn Rate'] * 100).round(2)

                fig = px.bar(
                    churn_by,
                    x=eng_col, y='Churn Rate %',
                    text='Churn Rate %',
                    title=f"Churn Rate by {eng_col}",
                    color='Churn Rate %',
                    color_continuous_scale='RdYlGn_r',
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            st.write("**Exited Distribution:**")
            st.dataframe(
                exited_dist.reset_index().rename(columns={'Exited': 'Label', 'count': 'Count'}),
                use_container_width=True,
            )
        else:
            st.warning(f"'{target}' column not found in dataset")