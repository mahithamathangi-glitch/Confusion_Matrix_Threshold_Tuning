import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Confusion Matrix & Threshold Tuning",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Confusion Matrix and Threshold Tuning")

st.write(
    """
    This application demonstrates how changing the classification
    probability threshold affects accuracy, precision, recall and F1 score.
    """
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    data = load_breast_cancer(as_frame=True)

    X = data.data

    # Original sklearn:
    # 0 = malignant
    # 1 = benign

    # Convert:
    # 0 = benign
    # 1 = malignant

    y = (data.target == 0).astype(int)

    return X, y


X, y = load_data()


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICT PROBABILITIES
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Threshold Settings")

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader(
    f"Confusion Matrix — Threshold {threshold:.2f}"
)

cm = confusion_matrix(
    y_test,
    predictions
)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)


# --------------------------------------------------
# THRESHOLD ANALYSIS
# --------------------------------------------------

st.subheader("Threshold Analysis")

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []

for t in thresholds:

    pred = (
        probabilities >= t
    ).astype(int)

    results.append({
        "Threshold": round(t, 2),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            pred,
            zero_division=0
        )
    })


results_df = pd.DataFrame(results)


st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# PRECISION / RECALL GRAPH
# --------------------------------------------------

st.subheader("Precision vs Recall")

fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

ax2.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

ax2.set_xlabel("Threshold")
ax2.set_ylabel("Score")
ax2.set_title("Precision vs Recall at Different Thresholds")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

st.info(
    """
    Lowering the threshold generally makes the model more likely to
    classify an observation as positive. This can increase recall but
    may reduce precision. Increasing the threshold generally has the
    opposite effect.
    """
)