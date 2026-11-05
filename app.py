"""Titanic Survival Predictor - Streamlit frontend.

Run:
    python train_model.py      # once, creates models/titanic_model.joblib
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from titanic_utils import add_features

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "titanic_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "Titanic-Dataset.csv"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {background: linear-gradient(120deg,#0b2545,#13315c 60%,#1d5c8a);
             padding: 1.6rem 2rem; border-radius: 14px; color: #fff; margin-bottom: 1.2rem;}
      .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
      .hero p {margin: .3rem 0 0; opacity: .85;}
      .result-ok  {background:#e6f6ec; border-left:6px solid #1a9850; padding:1rem 1.2rem; border-radius:8px; color:#0f3d22;}
      .result-bad {background:#fdeaea; border-left:6px solid #d73027; padding:1rem 1.2rem; border-radius:8px; color:#5a1410;}
      .result-ok h2, .result-bad h2 {margin:0 0 .2rem; color: inherit;}
    </style>
    <div class="hero">
      <h1>🚢 Titanic Survival Predictor</h1>
      <p>Enter passenger details and a machine-learning model estimates the chance of survival.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first, then reload this page.")
    st.stop()

model = load_model()
metrics = load_metrics() if METRICS_PATH.exists() else None

tab_predict, tab_explore, tab_model = st.tabs(["🔮 Predict", "📊 Explore Data", "🧠 Model Performance"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Passenger details")
        c1, c2 = st.columns(2)
        sex = c1.selectbox("Sex", ["female", "male"])
        pclass = c2.selectbox("Passenger class", [1, 2, 3], index=2,
                              format_func=lambda x: {1: "1st (Upper)", 2: "2nd (Middle)", 3: "3rd (Lower)"}[x])
        age = st.slider("Age", 0, 80, 29)
        c3, c4 = st.columns(2)
        sibsp = c3.number_input("Siblings / spouses aboard", 0, 8, 0)
        parch = c4.number_input("Parents / children aboard", 0, 6, 0)
        fare = st.slider("Ticket fare (£)", 0.0, 520.0, 32.0, step=0.5)
        embarked = st.selectbox(
            "Port of embarkation", ["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
        )
        go = st.button("Predict survival", type="primary", use_container_width=True)

    with right:
        st.subheader("Result")
        if go:
            row = pd.DataFrame([{
                "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
                "Parch": parch, "Fare": fare, "Embarked": embarked,
            }])
            proba = float(model.predict_proba(add_features(row))[0, 1])
            survived = proba >= 0.5
            css, title = ("result-ok", "Likely to survive") if survived else ("result-bad", "Unlikely to survive")
            st.markdown(
                f'<div class="{css}"><h2>{title}</h2>'
                f'Estimated survival probability: <b>{proba:.0%}</b></div>',
                unsafe_allow_html=True,
            )
            st.progress(proba)
            st.caption("Based on patterns in historical passenger data. This is a statistical estimate, not a certainty.")
        else:
            st.info("Fill in the details on the left and click **Predict survival**.")

# ---------------------------------------------------------------- Explore
with tab_explore:
    if not DATA_PATH.exists():
        st.warning("Place `Titanic-Dataset.csv` next to app.py to enable this tab.")
    else:
        df = load_data()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Passengers", f"{len(df):,}")
        k2.metric("Survival rate", f"{df['Survived'].mean():.1%}")
        k3.metric("Median age", f"{df['Age'].median():.0f}")
        k4.metric("Median fare", f"£{df['Fare'].median():.1f}")

        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.barplot(data=df, x="Sex", y="Survived", ax=ax, errorbar=None)
            ax.set_title("Survival rate by sex"); ax.set_ylabel("Survival rate")
            st.pyplot(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.barplot(data=df, x="Pclass", y="Survived", ax=ax, errorbar=None)
            ax.set_title("Survival rate by class"); ax.set_ylabel("Survival rate")
            st.pyplot(fig)

        c, d = st.columns(2)
        with c:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(data=df, x="Age", hue="Survived", kde=True, ax=ax)
            ax.set_title("Age distribution by survival")
            st.pyplot(fig)
        with d:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation heatmap")
            st.pyplot(fig)

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------- Model
with tab_model:
    if metrics is None:
        st.warning("metrics.json not found. Run `python train_model.py`.")
    else:
        st.success(f"Deployed model: **{metrics['best_model']}**")
        table = pd.DataFrame(metrics["results"]).T.drop(columns=["confusion_matrix"])
        st.dataframe(table, use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.markdown("**Top features**")
            imp = pd.DataFrame(metrics["feature_importances"], columns=["feature", "importance"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=imp, y="feature", x="importance", ax=ax)
            st.pyplot(fig)
        with b:
            st.markdown("**Confusion matrix (test set)**")
            cm = metrics["results"][metrics["best_model"]]["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)