# page_prediction.py - FINAL FIXED VERSION

import streamlit as st
import pandas as pd
import numpy as np
from src_auth import get_current_user_id, require_authentication
from database import get_user_models, get_user_datasets, save_prediction
from utils import load_csv
from feature_engineering import apply_feature_engineering
import json
import joblib


# ── Scenario definitions ───────────────────────────────────────────────────────

_SCENARIOS = {
    "— Enter manually —": {},
    "🔴 High Risk — Inactive German customer (Maria, 47)": {
        'CreditScore': 400.0, 'Age': 47.0, 'Tenure': 1.0,
        'Balance': 125000.0, 'NumOfProducts': 1.0,
        'HasCrCard': 0.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 60000.0,
        'Geography': 'Germany', 'Gender': 'Female',
        'story': (
            "Maria is a 47-year-old German customer with a poor credit score, "
            "high balance sitting idle, only 1 product, no credit card, and has "
            "been completely inactive. She joined just 1 year ago and shows every "
            "warning sign of someone about to leave the bank."
        ),
    },
    "🔴 High Risk — Overloaded German customer (Hans, 52)": {
        'CreditScore': 620.0, 'Age': 52.0, 'Tenure': 2.0,
        'Balance': 180000.0, 'NumOfProducts': 4.0,
        'HasCrCard': 1.0, 'IsActiveMember': 0.0, 'EstimatedSalary': 45000.0,
        'Geography': 'Germany', 'Gender': 'Male',
        'story': (
            "Hans is 52, German, inactive, and has 4 products — research shows "
            "customers with 3–4 products actually churn MORE than those with 2. "
            "He has a very high balance but low salary suggesting financial stress, "
            "and has only been a customer for 2 years. Inactivity + Germany + "
            "overloaded products makes him critically high risk."
        ),
    },
    "🟢 Low Risk — Loyal French customer (Pierre, 34)": {
        'CreditScore': 780.0, 'Age': 34.0, 'Tenure': 9.0,
        'Balance': 0.0, 'NumOfProducts': 2.0,
        'HasCrCard': 1.0, 'IsActiveMember': 1.0, 'EstimatedSalary': 135000.0,
        'Geography': 'France', 'Gender': 'Male',
        'story': (
            "Pierre is a 34-year-old French customer with excellent credit, "
            "9 years of tenure, 2 products, a credit card, and actively uses "
            "his account every month. Zero-balance customers in France are the "
            "most loyal segment — Pierre is a textbook retained customer."
        ),
    },
    "🟢 Low Risk — Young active Spanish customer (Sofia, 28)": {
        'CreditScore': 710.0, 'Age': 28.0, 'Tenure': 5.0,
        'Balance': 0.0, 'NumOfProducts': 2.0,
        'HasCrCard': 1.0, 'IsActiveMember': 1.0, 'EstimatedSalary': 88000.0,
        'Geography': 'Spain', 'Gender': 'Female',
        'story': (
            "Sofia is 28, Spanish, active, with good credit and 2 products. "
            "She has been a customer for 5 years and earns well. "
            "Young + active + 2 products is one of the strongest retention "
            "combinations regardless of geography."
        ),
    },
    "🟡 Borderline — Could go either way (Carmen, 41)": {
        'CreditScore': 620.0, 'Age': 41.0, 'Tenure': 4.0,
        'Balance': 95000.0, 'NumOfProducts': 1.0,
        'HasCrCard': 1.0, 'IsActiveMember': 1.0, 'EstimatedSalary': 72000.0,
        'Geography': 'Spain', 'Gender': 'Female',
        'story': (
            "Carmen is 41, Spanish, active but only has 1 product and a "
            "fair credit score. She has decent balance and has been a customer "
            "4 years. Her activity works in her favour but having only 1 product "
            "and fair credit keeps her at moderate risk. A small nudge — like "
            "offering a second product — could retain her long-term."
        ),
    },
}


# ── Factor impact definitions ─────────────────────────────────────────────────

def _get_factor_analysis(input_data):
    """Return list of (factor, signal_emoji, signal_label, explanation) for current inputs."""
    analysis = []

    # IsActiveMember — strongest single signal
    is_active = int(input_data.get('IsActiveMember', 1))
    if is_active == 0:
        analysis.append(("IsActiveMember = Inactive", "🔴", "MAJOR RISK",
            "Inactive members are ~2.5x more likely to churn. "
            "This is the single strongest churn signal in the dataset."))
    else:
        analysis.append(("IsActiveMember = Active", "🟢", "STRONG PROTECTION",
            "Active members churn at less than half the rate of inactive ones. "
            "Keep this customer engaged."))

    # Geography
    geo = input_data.get('Geography', 'France')
    if geo == 'Germany':
        analysis.append((f"Geography = {geo}", "🔴", "HIGH RISK",
            "Germany has the highest churn rate (~32%) in this dataset — "
            "nearly double that of France. German customers need extra attention."))
    elif geo == 'France':
        analysis.append((f"Geography = {geo}", "🟢", "LOW RISK",
            "France has the lowest churn rate (~16%) in the dataset. "
            "French customers are the most loyal segment."))
    else:
        analysis.append((f"Geography = {geo}", "🟡", "MODERATE",
            "Spain sits between France and Germany with ~17% churn rate. "
            "Slightly elevated but manageable risk."))

    # NumOfProducts
    n = min(int(input_data.get('NumOfProducts', 2)), 4)
    if n == 1:
        analysis.append((f"NumOfProducts = {n}", "🔴", "RISK",
            "Single-product customers churn most often — they have low "
            "switching cost and little reason to stay. Cross-selling is key."))
    elif n == 2:
        analysis.append((f"NumOfProducts = {n}", "🟢", "OPTIMAL",
            "2 products is the sweet spot — lowest churn rate in the dataset. "
            "The customer is engaged but not overloaded."))
    else:
        analysis.append((f"NumOfProducts = {n}", "🔴", "RISK",
            f"{n} products can mean over-commitment and dissatisfaction. "
            "Customers with 3–4 products surprisingly churn more than those with 2."))

    # Age
    age = input_data.get('Age', 35)
    if age < 35:
        analysis.append((f"Age = {age:.0f}", "🟢", "LOW RISK",
            "Younger customers (under 35) tend to stay longer. "
            "They are still building financial relationships."))
    elif age <= 50:
        analysis.append((f"Age = {age:.0f}", "🟡", "MODERATE",
            "Mid-age customers (35–50) show moderate churn tendency. "
            "They may be evaluating competitors more actively."))
    else:
        analysis.append((f"Age = {age:.0f}", "🔴", "ELEVATED RISK",
            "Customers over 50 churn more in this dataset — "
            "possibly consolidating finances or switching to premium banks."))

    # CreditScore
    cs = input_data.get('CreditScore', 650)
    if cs < 600:
        analysis.append((f"CreditScore = {cs:.0f}", "🔴", "RISK",
            "Poor credit score (below 600) correlates with higher churn. "
            "These customers may be under financial stress."))
    elif cs < 700:
        analysis.append((f"CreditScore = {cs:.0f}", "🟡", "MODERATE",
            "Fair credit (600–700) — average risk. "
            "Not a strong signal either way."))
    else:
        analysis.append((f"CreditScore = {cs:.0f}", "🟢", "PROTECTION",
            "Good credit score (700+) correlates with loyalty. "
            "Financially stable customers tend to stay."))

    # Balance
    bal = input_data.get('Balance', 0)
    if bal == 0:
        analysis.append(("Balance = €0", "🟢", "PROTECTION",
            "Zero-balance customers churn less — they use other products "
            "and services actively. This is a counterintuitive but strong signal."))
    elif bal < 100_000:
        analysis.append((f"Balance = €{bal:,.0f}", "🟡", "MODERATE",
            "Moderate balance — average churn risk. "
            "Not a strong signal in either direction."))
    else:
        analysis.append((f"Balance = €{bal:,.0f}", "🔴", "ELEVATED RISK",
            f"Very high balance (€{bal:,.0f}) customers churn more — "
            "they may be attractive targets for premium competitor offers."))

    # Tenure
    tenure = input_data.get('Tenure', 5)
    if tenure <= 2:
        analysis.append((f"Tenure = {tenure:.0f} years", "🔴", "RISK",
            "New customers (0–2 years) haven't built deep loyalty yet. "
            "Early engagement is critical to retain them."))
    elif tenure <= 6:
        analysis.append((f"Tenure = {tenure:.0f} years", "🟡", "MODERATE",
            "Established customer (3–6 years) — not deeply committed yet "
            "but has some history with the bank."))
    else:
        analysis.append((f"Tenure = {tenure:.0f} years", "🟢", "STRONG PROTECTION",
            f"{tenure:.0f} years is long-term loyalty. "
            "Customers with 7+ years rarely churn unless something goes very wrong."))

    return analysis


# ── Helper functions ───────────────────────────────────────────────────────────

def load_model_with_preprocessor(model_path):
    try:
        package = joblib.load(model_path)
        if isinstance(package, dict):
            return (
                package.get("model"),
                package.get("preprocessor"),
                package.get("feature_columns"),
                package.get("selected_features"),
            )
        st.error("❌ Invalid model format — expected a dict package")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None


def get_raw_input_cols(df):
    drop = {'rownumber', 'customerid', 'surname', 'exited', 'churn', 'churned'}
    return [c for c in df.columns if c.lower() not in drop]


def _apply_engineering_and_select(raw_input_df, selected_features):
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df = raw_input_df.drop(columns=[c for c in drop_cols if c in raw_input_df.columns])
    engineered = apply_feature_engineering(df)
    missing = set(selected_features) - set(engineered.columns)
    if missing:
        st.warning(f"⚠️ Features missing after engineering (filled with 0): {missing}")
    return engineered.reindex(columns=selected_features, fill_value=0)


def _run_preprocessor(preprocessor, df):
    result = preprocessor.preprocess(
        df,
        is_training=False,
        missing_method='mean',
        encoding_method='label',
        scaling_method='standard',
        create_features=False,
    )
    return result[0] if isinstance(result, tuple) else result


def get_risk_level(prediction: int, confidence: float) -> str:
    if prediction == 0:
        if confidence >= 0.80: return "🟢 Very Low"
        if confidence >= 0.70: return "🟡 Low"
        return "🟠 Moderate"
    else:
        if confidence >= 0.90: return "🔴 Critical"
        if confidence >= 0.80: return "🔴 Very High"
        if confidence >= 0.70: return "🟠 High"
        if confidence >= 0.60: return "🟡 Medium"
        return "🟢 Low"


# ── Main page ──────────────────────────────────────────────────────────────────

def show_prediction_page():
    st.title("🎯 Make Predictions")
    st.markdown("Use trained models to predict customer churn")

    require_authentication()
    user_id = get_current_user_id()

    models   = get_user_models(user_id)
    datasets = get_user_datasets(user_id)

    if not models:
        st.warning("❌ No trained models found. Please train a model first.")
        return
    if not datasets:
        st.warning("❌ No datasets found. Please upload a dataset first.")
        return

    model_names = {
        f"{m['model_name']} (Accuracy: {m.get('accuracy', 0):.2%})": m
        for m in models
    }
    selected_model_label = st.selectbox("Select Model", list(model_names.keys()))
    selected_model       = model_names[selected_model_label]

    model, preprocessor, feature_columns, selected_features = \
        load_model_with_preprocessor(selected_model['model_path'])

    if model is None:
        st.error("❌ Failed to load model")
        return
    if not selected_features or not feature_columns:
        st.error("❌ Model is missing feature metadata. Please re-train the model.")
        return

    dataset_names          = {d['dataset_name']: d['file_path'] for d in datasets}
    selected_dataset_label = st.selectbox(
        "Select Dataset (for feature reference)", list(dataset_names.keys())
    )
    raw_df = load_csv(dataset_names[selected_dataset_label])

    if raw_df is None:
        st.error("❌ Failed to load dataset")
        return

    st.divider()
    prediction_type = st.radio(
        "Prediction Type", ["Single Prediction", "Batch Prediction"], horizontal=True
    )

    if prediction_type == "Single Prediction":
        show_single_prediction(
            user_id, model, preprocessor, feature_columns,
            selected_features, raw_df, selected_model
        )
    else:
        show_batch_prediction(
            user_id, model, preprocessor, feature_columns,
            selected_features, raw_df, selected_model
        )


# ── Single prediction ──────────────────────────────────────────────────────────

def show_single_prediction(user_id, model, preprocessor, feature_columns,
                           selected_features, raw_df, selected_model):
    st.subheader("📝 Enter Customer Information")

    # ── Scenario picker ───────────────────────────────────────────────────────
    st.markdown("**💡 Load a sample scenario or enter values manually:**")
    chosen_scenario = st.selectbox(
        "Sample Scenarios", list(_SCENARIOS.keys()), key="scenario_select"
    )
    scenario_data = _SCENARIOS[chosen_scenario]

    if chosen_scenario != "— Enter manually —":
        st.info(f"📖 **Story:** {scenario_data['story']}")

    raw_input_cols   = get_raw_input_cols(raw_df)
    numeric_cols     = [c for c in raw_input_cols
                        if pd.api.types.is_numeric_dtype(raw_df[c])]
    categorical_cols = [c for c in raw_input_cols
                        if not pd.api.types.is_numeric_dtype(raw_df[c])]

    input_data = {}

    # ── Categorical dropdowns ─────────────────────────────────────────────────
    if categorical_cols:
        st.markdown("**🧍 Demographics**")
        cat_ui = st.columns(min(len(categorical_cols), 3))
        for i, col in enumerate(categorical_cols):
            with cat_ui[i % 3]:
                unique_vals = sorted(raw_df[col].dropna().unique().tolist())
                default_val = scenario_data.get(col, unique_vals[0])
                default_idx = unique_vals.index(default_val) \
                              if default_val in unique_vals else 0
                input_data[col] = st.selectbox(
                    col, unique_vals, index=default_idx, key=f"sel_{col}"
                )

    st.divider()

    # ── Numeric sliders ───────────────────────────────────────────────────────
    st.markdown("**🏦 Account Details**")
    col_widgets = st.columns(2)

    for i, col in enumerate(numeric_cols):
        with col_widgets[i % 2]:
            min_v     = float(raw_df[col].min())
            max_v     = float(raw_df[col].max())
            mean_v    = float(raw_df[col].mean())
            default_v = float(np.clip(
                scenario_data.get(col, mean_v), min_v, max_v
            ))
            step = max((max_v - min_v) / 100, 0.01)

            st.markdown(f"**{col}** — `{default_v:,.2f}`")
            input_data[col] = st.slider(
                col,
                min_value=min_v, max_value=max_v,
                value=default_v, step=step,
                label_visibility="collapsed",
                key=f"num_{col}",
            )

    # ── Live factor analysis ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 How These Values Affect Churn Risk")
    st.caption("Updates live as you adjust the values above.")

    analysis = _get_factor_analysis(input_data)

    # Count risk signals
    red_count  = sum(1 for _, sig, _, _ in analysis if sig == "🔴")
    green_count = sum(1 for _, sig, _, _ in analysis if sig == "🟢")

    col_r, col_g, col_n = st.columns(3)
    with col_r: st.metric("🔴 Risk Factors",       red_count)
    with col_g: st.metric("🟢 Protective Factors", green_count)
    with col_n: st.metric("🟡 Neutral Factors",    len(analysis) - red_count - green_count)

    st.markdown("")
    for factor, emoji, label, explanation in analysis:
        st.markdown(
            f"{emoji} **{factor}** &nbsp;·&nbsp; *{label}*  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{explanation}"
        )

    st.divider()

    if st.button("🔮 Make Prediction", use_container_width=True, key="predict_btn"):
        try:
            raw_input_df  = pd.DataFrame([input_data])
            engineered_df = _apply_engineering_and_select(raw_input_df, selected_features)
            X_processed   = _run_preprocessor(preprocessor, engineered_df)
            X_processed   = X_processed.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(X_processed)[0]
            proba      = model.predict_proba(X_processed)[0] \
                         if hasattr(model, 'predict_proba') else [0.5, 0.5]
            confidence  = float(max(proba))
            churn_proba = float(proba[1])

            save_prediction(
                user_id, selected_model['model_id'],
                json.dumps(input_data), int(prediction), float(confidence)
            )

            # ── Results ───────────────────────────────────────────────────────
            st.divider()
            st.subheader("🎯 Prediction Results")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if prediction == 1:
                    st.error("🔴 Likely to Churn")
                else:
                    st.success("🟢 Likely to Retain")
            with c2:
                st.metric("Churn Probability",  f"{churn_proba:.2%}")
            with c3:
                st.metric("Retain Probability", f"{proba[0]:.2%}")
            with c4:
                st.metric("Risk Level", get_risk_level(prediction, confidence))

            st.progress(float(churn_proba),
                        text=f"Churn risk: {churn_proba:.2%}")

            # ── Scenario-tied explanation ─────────────────────────────────────
            st.divider()
            st.subheader("🔍 Why did the model predict this?")

            # Top risk drivers
            risk_factors      = [(f, e) for f, s, _, e in analysis if s == "🔴"]
            protective_factors = [(f, e) for f, s, _, e in analysis if s == "🟢"]

            if prediction == 1:
                st.error(
                    f"The model flagged **{red_count} risk factor(s)** "
                    f"and only {green_count} protective factor(s) for this customer."
                )
                if risk_factors:
                    st.markdown("**🔴 Key reasons driving churn prediction:**")
                    for factor, exp in risk_factors:
                        st.markdown(f"- **{factor}**: {exp}")
                if protective_factors:
                    st.markdown("**🟢 Factors that reduced the risk score:**")
                    for factor, exp in protective_factors:
                        st.markdown(f"- **{factor}**: {exp}")
            else:
                st.success(
                    f"The model found **{green_count} protective factor(s)** "
                    f"outweighing {red_count} risk factor(s) for this customer."
                )
                if protective_factors:
                    st.markdown("**🟢 Key reasons driving retention prediction:**")
                    for factor, exp in protective_factors:
                        st.markdown(f"- **{factor}**: {exp}")
                if risk_factors:
                    st.markdown("**🔴 Factors that still pose some risk:**")
                    for factor, exp in risk_factors:
                        st.markdown(f"- **{factor}**: {exp}")

            # ── Actionable recommendation ─────────────────────────────────────
            st.divider()
            st.subheader("💡 Recommended Actions")

            if prediction == 1:
                st.warning("This customer is at risk of churning. Suggested actions:")
                # Tailor actions to the specific risk factors found
                if any("Inactive" in f for f, _ in risk_factors):
                    st.markdown("- 📲 **Re-engagement campaign** — send personalised "
                                "activity incentives immediately.")
                if any("Germany" in f for f, _ in risk_factors):
                    st.markdown("- 🇩🇪 **Germany retention programme** — offer "
                                "region-specific loyalty rewards.")
                if any("NumOfProducts = 1" in f for f, _ in risk_factors):
                    st.markdown("- 🛍️ **Cross-sell a second product** — customers "
                                "with 2 products churn at much lower rates.")
                if any("Balance" in f and "High" in e for f, e in risk_factors):
                    st.markdown("- 💰 **High-value customer care** — assign a "
                                "dedicated relationship manager.")
                if any("Age" in f for f, _ in risk_factors):
                    st.markdown("- 👴 **Senior customer support** — offer simplified "
                                "services and dedicated phone support.")
                st.markdown("- 📞 **Personal outreach** within 7 days.")
                st.markdown("- 💸 **Retention offer** — discount or fee waiver.")
            else:
                st.info("This customer is likely to stay. Suggested actions:")
                if any("NumOfProducts = 2" in f for f, _ in protective_factors):
                    st.markdown("- 🎁 **Upsell opportunity** — they already have 2 "
                                "products; consider a premium upgrade.")
                if any("Active" in f for f, _ in protective_factors):
                    st.markdown("- 📈 **Leverage engagement** — active customers "
                                "respond well to loyalty rewards.")
                if any("Tenure" in f for f, _ in protective_factors):
                    st.markdown("- 🏆 **Loyalty recognition** — acknowledge their "
                                "long relationship with a milestone reward.")
                st.markdown("- 📧 **Regular satisfaction check-in** every quarter.")
                st.markdown("- 👍 **Maintain service quality** — don't take "
                            "loyal customers for granted.")

        except Exception as e:
            import traceback
            st.error(f"❌ Error making prediction: {str(e)}")
            st.code(traceback.format_exc())


# ── Batch prediction ───────────────────────────────────────────────────────────

def show_batch_prediction(user_id, model, preprocessor, feature_columns,
                          selected_features, raw_df, selected_model):
    st.subheader("📤 Upload CSV for Batch Prediction")
    st.info(
        "Upload a CSV with the **raw** columns (CreditScore, Age, Geography …). "
        "Do **not** include RowNumber, CustomerId, Surname, or the target column. "
        "Feature engineering is applied automatically."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")
    if not uploaded_file:
        return

    df_input = pd.read_csv(uploaded_file)
    st.write("**Preview (first 5 rows):**")
    st.dataframe(df_input.head(), use_container_width=True)

    if st.button("🔮 Make Batch Predictions", use_container_width=True, key="batch_predict_btn"):
        try:
            engineered_df = _apply_engineering_and_select(df_input, selected_features)
            X_processed   = _run_preprocessor(preprocessor, engineered_df)
            X_processed   = X_processed.reindex(columns=feature_columns, fill_value=0)

            predictions = model.predict(X_processed)
            if hasattr(model, 'predict_proba'):
                probas       = model.predict_proba(X_processed)
                confidences  = probas.max(axis=1)
                churn_probas = probas[:, 1]
            else:
                confidences  = np.ones(len(predictions)) * 0.5
                churn_probas = confidences

            results = pd.DataFrame({
                "prediction":        predictions.astype(int),
                "prediction_label":  ["Churn" if p == 1 else "Retain" for p in predictions],
                "churn_probability": [f"{p:.2%}" for p in churn_probas],
                "confidence":        [f"{c:.2%}" for c in confidences],
                "risk_level":        [get_risk_level(p, c)
                                      for p, c in zip(predictions, confidences)],
            })

            display_input = df_input.drop(
                columns=[c for c in ['Exited', 'Churn', 'RowNumber', 'CustomerId', 'Surname']
                         if c in df_input.columns]
            )
            output_df = pd.concat(
                [display_input.reset_index(drop=True), results.reset_index(drop=True)],
                axis=1
            )

            st.success("✅ Predictions completed!")
            st.dataframe(output_df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Customers",  len(predictions))
            with c2: st.metric("Predicted Churn",  int((predictions == 1).sum()))
            with c3: st.metric("Predicted Retain", int((predictions == 0).sum()))
            with c4: st.metric("Avg Churn Prob",   f"{churn_probas.mean():.2%}")

            st.download_button(
                "📥 Download Results",
                output_df.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )

        except Exception as e:
            import traceback
            st.error(f"❌ Error making predictions: {str(e)}")
            st.code(traceback.format_exc())