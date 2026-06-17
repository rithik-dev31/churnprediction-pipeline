"""
Settings and preferences page.
"""

import streamlit as st
from src_auth import get_current_user_id, get_current_username, require_authentication
from database import get_user
import platform
import streamlit as _st_module


def show_settings_page():
    """Display settings page."""
    st.title("⚙️ Settings")
    st.markdown("Manage your preferences and account settings")

    require_authentication()
    user_id  = get_current_user_id()
    username = get_current_username()

    user_info = get_user(user_id)
    if not user_info:
        st.error("Could not load user information")
        return

    tab1, tab2, tab3 = st.tabs(["👤 Profile", "🎨 Preferences", "ℹ️ About"])

    # ── Tab 1: Profile ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("👤 Profile Information")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Username**")
            st.code(user_info['username'], language=None)
        with col2:
            st.markdown("**Email**")
            st.code(user_info['email'], language=None)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Account Created**")
            st.code(str(user_info['created_at']), language=None)
        with col2:
            st.markdown("**User ID**")
            st.code(str(user_info['user_id']), language=None)

        st.divider()

        # Account stats
        st.subheader("📊 Account Summary")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("👤 Username", user_info['username'])
        with m2:
            # Show days since account creation if parseable, else just show date
            try:
                from datetime import datetime
                created = datetime.strptime(
                    str(user_info['created_at'])[:19], "%Y-%m-%d %H:%M:%S"
                )
                days = (datetime.now() - created).days
                st.metric("📅 Member For", f"{days} days")
            except Exception:
                st.metric("📅 Created", str(user_info['created_at'])[:10])
        with m3:
            st.metric("🔑 Account Status", "Active")

    # ── Tab 2: Preferences ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("🎨 Preferences")
        st.caption(
            "These preferences are saved for your current session. "
            "They reset when you close the browser."
        )

        # ── Appearance ────────────────────────────────────────────────────────
        st.markdown("#### 🖥️ Appearance")
        col1, col2 = st.columns(2)
        with col1:
            theme = st.radio(
                "Application Theme",
                ["Light", "Dark", "Auto"],
                index=["Light", "Dark", "Auto"].index(
                    st.session_state.get("pref_theme", "Auto")
                ),
                horizontal=True,
                key="theme_select",
                help="Visual theme for the application.",
            )
        with col2:
            notifications = st.checkbox(
                "Enable Notifications",
                value=st.session_state.get("pref_notifications", True),
                key="notifications",
                help="Show success/warning notifications across pages.",
            )

        st.divider()

        # ── Model Training ────────────────────────────────────────────────────
        st.markdown("#### 🤖 Model Training")
        col1, col2 = st.columns(2)
        with col1:
            auto_save = st.checkbox(
                "Auto-save Models after Training",
                value=st.session_state.get("pref_auto_save", True),
                help="Automatically save every trained model to disk.",
            )
        with col2:
            show_warnings = st.checkbox(
                "Show Training Warnings",
                value=st.session_state.get("pref_show_warnings", True),
                help="Display sklearn and data warnings during training.",
            )

        st.divider()

        # ── Data Display ──────────────────────────────────────────────────────
        st.markdown("#### 📋 Data Display")
        st.caption(
            "Controls how tables and numbers appear across the app — "
            "EDA, predictions, and model results."
        )
        col1, col2 = st.columns(2)
        with col1:
            rows_per_page = st.slider(
                "Rows Per Page",
                min_value=5, max_value=100,
                value=st.session_state.get("pref_rows_per_page", 20),
                step=5,
                help="How many rows to show in data preview tables.",
            )
            st.caption(f"Tables will show **{rows_per_page} rows** at a time.")
        with col2:
            decimal_places = st.slider(
                "Decimal Places",
                min_value=1, max_value=6,
                value=st.session_state.get("pref_decimal_places", 4),
                help="Decimal precision for numeric values in results tables.",
            )
            st.caption(f"Numbers will show **{decimal_places} decimal places**.")

        st.divider()

        # ── Save button ───────────────────────────────────────────────────────
        if st.button("💾 Save Preferences", use_container_width=True):
            st.session_state["pref_theme"]          = theme
            st.session_state["pref_notifications"]  = notifications
            st.session_state["pref_auto_save"]      = auto_save
            st.session_state["pref_show_warnings"]  = show_warnings
            st.session_state["pref_rows_per_page"]  = rows_per_page
            st.session_state["pref_decimal_places"] = decimal_places
            st.success("✅ Preferences saved for this session!")

        # ── Current active preferences preview ───────────────────────────────
        with st.expander("👁️ View Current Active Preferences"):
            st.json({
                "theme":          st.session_state.get("pref_theme", "Auto"),
                "notifications":  st.session_state.get("pref_notifications", True),
                "auto_save":      st.session_state.get("pref_auto_save", True),
                "show_warnings":  st.session_state.get("pref_show_warnings", True),
                "rows_per_page":  st.session_state.get("pref_rows_per_page", 20),
                "decimal_places": st.session_state.get("pref_decimal_places", 4),
            })

    # ── Tab 3: About ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("ℹ️ About This Application")

        # App identity
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 App Version",       "1.0.0")
        with col2:
            st.metric("🐍 Python Version",    platform.python_version())
        with col3:
            st.metric("⚡ Streamlit Version", _st_module.__version__)

        st.divider()

        # Description
        st.markdown("### 🧠 Customer Churn Prediction System")
        st.markdown(
            "A full machine learning pipeline for predicting customer churn "
            "using multiple algorithms, advanced preprocessing, and class "
            "imbalance handling."
        )

        st.divider()

        # Features in two columns
        st.markdown("### ✨ Features")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("""
- 🔐 Secure authentication system
- 📊 Interactive data exploration (EDA)
- 🤖 6 ML models with comparison
- ⚖️ Class imbalance handling (SMOTE, etc.)
- 🧬 Automated feature engineering
""")
        with f2:
            st.markdown("""
- 📈 Real-time single & batch predictions
- 💾 Model persistence and management
- 📋 Comprehensive evaluation metrics
- 🔍 Scenario-based prediction explanations
- 📥 Downloadable results
""")

        st.divider()

        # Models
        st.markdown("### 🤖 Supported Models")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("1. Logistic Regression\n2. Random Forest")
        with m2:
            st.markdown("3. Decision Tree\n4. XGBoost")
        with m3:
            st.markdown("5. Support Vector Machine\n6. K-Nearest Neighbors")

        st.divider()

        # Metrics
        st.markdown("### 📊 Evaluation Metrics")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("- Accuracy\n- Precision\n- Recall\n- F1-Score")
        with e2:
            st.markdown("- ROC-AUC\n- Confusion Matrix\n- Cross-Validation scores")

        st.divider()

        # Tech stack
        st.markdown("### 🛠️ Technologies Used")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**Backend**\n- Python\n- scikit-learn\n- XGBoost\n- imbalanced-learn")
        with t2:
            st.markdown("**Frontend**\n- Streamlit\n- Plotly\n- Matplotlib\n- Seaborn")
        with t3:
            st.markdown("**Storage**\n- SQLite\n- joblib\n- pandas")

        st.divider()

        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.info(
            "1. **Dataset** — must include `Exited` or `Churn` as the target column.\n"
            "2. **Feature Engineering** — applied automatically before training and prediction.\n"
            "3. **Imbalance** — use SMOTE (recommended) for imbalanced churn datasets.\n"
            "4. **Model Selection** — compare F1-Score, not just Accuracy, for imbalanced data.\n"
            "5. **Prediction** — always use the same dataset the model was trained on as reference."
        )

        st.divider()
        st.caption("**Author:** ML Engineering Team &nbsp;|&nbsp; **License:** MIT")