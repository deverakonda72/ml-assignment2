"""
naive_bayes.py  -  Naive Bayes (Gaussian), step by step
=======================================================
This python script implements the naive bayes logic for classification
This generates the  required dataset on its own.
It is structured to:
   - Generate required dataset
   - clears any previously generated models ( Optional)
   - builds the model with the generated synthetic data
   - trains the model with the generated synthetic data
   - tests the models

 Creates the respective folders for:
   / model
   / data
   / plots ( confusion matrix , roc curves
"""
import argparse
import os
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files (works even without a screen)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.naive_bayes import GaussianNB

from generate_data import TARGET_COLUMN, generate_dataset, save_dataset

##################################################
# configuration settings for model directories, data directories
##################################################
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "model")
PLOT_DIR = os.path.join(HERE, "plots")

DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes.joblib")

RANDOM_STATE = 42  # same seed everywhere, so results can be reproduced


class NaiveBayesModel:
    MODEL_NAME = "Naive Bayes (Gaussian)"

    ##################################################
    # Initializes the KNN Model
    ##################################################
    def __init__(self, n_rows=2000, n_classes=4, test_size=0.2, split_method="stratified"):
        self.n_rows = n_rows
        self.n_classes = n_classes
        self.test_size = test_size
        self.split_method = split_method

    ##################################################
    #BUILDS THE MODEL, USES SCEKITLEARN LIB
    ##################################################
    def build_model(self):
        # No scaling needed: the model learns a mean and variance per feature.
        # var_smoothing adds a tiny amount to every variance for numerical stability.
        return Pipeline([
            ("classifier", GaussianNB(var_smoothing=1e-9)),
        ])

    ##################################################
    # CLEANS THE EARLIER MODEL PARAMETERS
    ##################################################
    def clean(self):
        """Delete everything a previous run produced, so training starts fresh."""
        print(f"\n===== STEP 1: CLEAN ({self.MODEL_NAME}) =====")
        for folder in (MODEL_DIR, DATA_DIR, PLOT_DIR):
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Deleted {folder}")
            else:
                print(f"Nothing to delete at {folder}")

    ##################################################
    #TRAINS THE MODEL
    ##################################################
    def train(self):
        print(f"\n===== STEP 2: TRAIN ({self.MODEL_NAME}) =====")

        # 2a. Generate the synthetic dataset and save it
        data = generate_dataset(self.n_rows, self.n_classes, RANDOM_STATE)
        save_dataset(data, DATASET_CSV)
        print(f"Generated {len(data)} rows, {data.shape[1] - 1} features, "
              f"{self.n_classes} classes -> {DATASET_CSV}")

        # 2b. Separate the inputs (X) from the answer we want to predict (y)
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]

        # 2c. Split into training rows and test rows
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 2d. Save both parts; the test step (and the Streamlit app) read the test file
        self.save_split(X_train, y_train, TRAIN_CSV)
        self.save_split(X_test, y_test, TEST_CSV)
        print(f"Train rows: {len(X_train)} -> {TRAIN_CSV}")
        print(f"Test rows : {len(X_test)} -> {TEST_CSV}")

        # 2e. Build the model and fit it on the training rows only
        model = self.build_model()
        model.fit(X_train, y_train)

        # 2f. Accuracy on the training rows. If this is much higher than the
        #     test accuracy you see in step 3, the model is overfitting.
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        print(f"Training accuracy: {train_accuracy:.4f}")

        # 2g. Save the trained model to disk
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def split_data(self, X, y):
        """
        Split rows into train and test sets.

        split_method = "stratified": every class keeps the same share in train
                                     and test (recommended for classification)
        split_method = "random":     plain random shuffle and cut
        """
        if self.split_method not in ("stratified", "random"):
            raise ValueError("split_method must be 'stratified' or 'random'")
        stratify = y if self.split_method == "stratified" else None
        print(f"Splitting data: method = {self.split_method}, test_size = {self.test_size}")
        return train_test_split(X, y, test_size=self.test_size, stratify=stratify,
                                random_state=RANDOM_STATE)

    @staticmethod
    def save_split(X, y, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part = X.copy()
        part[TARGET_COLUMN] = y
        part.to_csv(path, index=False)

    ##################################################
    # TESTS THE MODEL
    ##################################################
    def test(self, test_csv=TEST_CSV):
        print(f"\n===== STEP 3: TEST ({self.MODEL_NAME}) =====")

        # 3a. Load the model saved in step 2 (nothing is retrained here)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run the train step first.")
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model {MODEL_PATH}")

        # 3b. Load the test rows
        test_data = pd.read_csv(test_csv)
        X_test = test_data[model.feature_names_in_]
        y_test = test_data[TARGET_COLUMN]
        print(f"Loaded {len(test_data)} test rows from {test_csv}")

        # 3c. Predict: the class for each row, and the probability of every class
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        classes = model.classes_

        # 3d. Calculate and print the 6 evaluation metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba, classes)
        self.print_metrics(metrics)
        print("\nClassification report (per class):")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        # 3e. Save plots
        self.plot_confusion_matrix(y_test, y_pred, classes)
        self.plot_roc_curves(y_test, y_proba, classes)
        return metrics

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba, classes):
        """
        The 6 required metrics. With more than 2 classes:
          * Precision, Recall and F1 are averaged over classes, weighted by class size
          * AUC is "one-vs-rest": one ROC curve per class, then a weighted average
        """
        if len(classes) == 2:
            auc, average = roc_auc_score(y_true, y_proba[:, 1]), "binary"
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                                labels=classes)
            average = "weighted"
        return {
            "ACCURACY": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "PRECISION": precision_score(y_true, y_pred, average=average, zero_division=0),
            "RECALL": recall_score(y_true, y_pred, average=average, zero_division=0),
            "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

    def print_metrics(self, metrics):
        headers = ["Model Name"] + list(metrics)
        values = [self.MODEL_NAME] + [f"{v:.4f}" for v in metrics.values()]
        widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
        line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        row = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
        print("\nEvaluation metrics on the test data")
        print("\n".join([line, row(headers), line, row(values), line]))

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        os.makedirs(PLOT_DIR, exist_ok=True)
        matrix = confusion_matrix(y_true, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{self.MODEL_NAME}: confusion matrix")
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {path}")

    def plot_roc_curves(self, y_true, y_proba, classes):
        os.makedirs(PLOT_DIR, exist_ok=True)
        y_onehot = label_binarize(y_true, classes=classes)
        if y_onehot.shape[1] == 1:  # binary case
            y_onehot = np.hstack([1 - y_onehot, y_onehot])
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_proba[:, i])
            auc = roc_auc_score(y_onehot[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)  # a random guess
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{self.MODEL_NAME}: ROC curves (one-vs-rest)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {path}")


##################################################
# Command line: choose which step to run
##################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{NaiveBayesModel.MODEL_NAME}: clean, train, test")
    parser.add_argument("step", nargs="?", default="all", choices=["clean", "train", "test", "all"],
                        help="which step to run (default: all three in order)")
    parser.add_argument("--rows", type=int, default=2000, help="synthetic rows to generate")
    parser.add_argument("--classes", type=int, default=4, help="synthetic classes to generate")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of rows used for testing")
    parser.add_argument("--split", default="stratified", choices=["stratified", "random"],
                        help="train/test split method")
    parser.add_argument("--test-data", default=TEST_CSV, help="CSV to use in the test step")
    args = parser.parse_args()

    runner = NaiveBayesModel(n_rows=args.rows, n_classes=args.classes,
                       test_size=args.test_size, split_method=args.split)
    if args.step in ("clean", "all"):
        runner.clean()
    if args.step in ("train", "all"):
        runner.train()
    if args.step in ("test", "all"):
        runner.test(args.test_data)
