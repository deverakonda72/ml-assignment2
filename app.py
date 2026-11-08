"""
app.py - Streamlit app for the ML classification assignment.

Run from the project folder:
    streamlit run app.py

Each model lives in its own folder and is trained there first, for example:
    python logistic_regression/logistic_regression.py      (clean -> train -> test)

The app then:
    * lets you upload a test CSV (12 feature columns + "target").
      If nothing is uploaded, the bundled test_data.csv is used.
    * lets you pick a model from a dropdown. It loads <folder>/model/<name>.joblib.
    * shows Accuracy, AUC, Precision, Recall, F1 and MCC, the confusion matrix
      and the classification report, plus a comparison of all trained models.

The app never retrains on the uploaded data. It only loads the saved models
and calls predict / predict_proba.
"""
import os
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET_COLUMN = "target"

# Display name -> (model folder, model file name). Each folder trains its own model.
MODELS = {
    "Logistic Regression": ("logistic_regression", "logistic_regression.joblib"),
    "Decision Tree": ("decision_tree", "decision_tree.joblib"),
    "K-Nearest Neighbor": ("knn", "knn.joblib"),
    "Naive Bayes (Gaussian)": ("naive_bayes", "naive_bayes.joblib"),
    "Support Vector Machine": ("svm", "svm.joblib"),
}


# ---------------------------------------------------------------- helpers
def model_path(name):
    folder, file_name = MODELS[name]
    return os.path.join(BASE_DIR, folder, "model", file_name)


def script_path(name):
    folder, _ = MODELS[name]
    return os.path.join(BASE_DIR, folder, f"{folder}.py")


def trained_models():
    return [name for name in MODELS if os.path.exists(model_path(name))]


@st.cache_resource
def load_model(path, modified_time):
    # modified_time is part of the cache key, so a retrained model is reloaded
    return joblib.load(path)


def get_model(name):
    path = model_path(name)
    return load_model(path, os.path.getmtime(path))


def run_model_script(name):
    """Run clean -> train -> test in the model's own folder (training data only)."""
    return subprocess.run([sys.executable, script_path(name), "all"],
                          cwd=os.path.dirname(script_path(name)),
                          capture_output=True, text=True)


def read_test_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    if os.path.exists(DEFAULT_TEST_CSV):
        return pd.read_csv(DEFAULT_TEST_CSV), "test_data.csv (default)"
    return None, None


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Same 6 metrics as the model scripts (weighted averages, one-vs-rest AUC)."""
    if len(classes) == 2:
        auc, average = roc_auc_score(y_true, y_proba[:, 1]), "binary"
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes)
        average = "weighted"
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate(model, df):
    X = df[list(model.feature_names_in_)]
    y = df[TARGET_COLUMN]
    y_pred = model.predict(X)
    return y, y_pred, compute_metrics(y, y_pred, model.predict_proba(X), np.array(model.classes_))


def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- UI
st.set_page_config(page_title="ML Classification Models", layout="wide")
st.title("Multi-class classification: model evaluation")
st.caption("Evaluates saved models on test data. Models are not retrained on the uploaded file.")

available = trained_models()
missing_models = [name for name in MODELS if name not in available]

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV (12 features + target)", type=["csv"])
    if os.path.exists(DEFAULT_TEST_CSV):
        with open(DEFAULT_TEST_CSV, "rb") as f:
            st.download_button("Download sample test_data.csv", f, file_name="test_data.csv",
                               mime="text/csv")

    st.header("2. Model")
    if missing_models:
        st.warning("Not trained yet: " + ", ".join(missing_models))
        st.caption("Run each model's script in its folder, or use the button below.")
        if st.button("Train missing models"):
            for name in missing_models:
                with st.spinner(f"Training {name} ..."):
                    result = run_model_script(name)
                if result.returncode != 0:
                    st.error(f"{name} failed:\n{result.stderr[-1500:]}")
            st.rerun()
    if not available:
        st.stop()
    model_name = st.selectbox("Select model", available)

df, source = read_test_data(uploaded)
if df is None:
    st.info("Upload a test CSV to begin.")
    st.stop()

model = get_model(model_name)
features = list(model.feature_names_in_)
missing_cols = [c for c in features + [TARGET_COLUMN] if c not in df.columns]
if missing_cols:
    st.error(f"The CSV is missing required columns: {', '.join(missing_cols)}")
    st.stop()

classes = np.array(model.classes_)
y_test, y_pred, metrics = evaluate(model, df)

st.subheader(f"{model_name} on {source}")
st.write(f"{len(df)} rows, {len(features)} features, classes: {', '.join(map(str, classes))}. "
         f"Model file: `{os.path.relpath(model_path(model_name), BASE_DIR)}`")

cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

table = pd.DataFrame([{"Model Name": model_name, **{k.upper(): round(v, 4) for k, v in metrics.items()}}])
st.dataframe(table, hide_index=True, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion_matrix(y_test, y_pred, classes, f"{model_name}: confusion matrix"))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")

with st.expander("Compare all trained models on this data", expanded=True):
    rows = []
    for name in available:
        _, _, m = evaluate(get_model(name), df)
        rows.append({"Model Name": name, **{k.upper(): round(v, 4) for k, v in m.items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
