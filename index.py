#!/usr/bin/env python3

# =======================
# SILKY SKY AIRWAYS - TASK 1
# Logistic Regression + Random Forest for Passenger Satisfaction (Headless/CLI)
# =======================

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Optional: For reproducibility ---
import random
random.seed(42)
np.random.seed(42)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silky Sky Airways - Satisfaction Modeling (LogReg + RandomForest)",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to SILKYSKY_DATA_CW2.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save plots and reports (default: outputs)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    return parser.parse_args()


# -----------------------------
# Data & Preprocessing
# -----------------------------

def load_dataframe(csv_path: str) -> pd.DataFrame:
    # Fallback encodings to be resilient
    for enc in ("latin1", "utf-8", "cp1252"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            continue
    # Last attempt without encoding
    return pd.read_csv(csv_path)


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], str, List[str]]:
    # Expected columns (from original script)
    target_col = "Satisfied"
    drop_cols = [col for col in ["Ref", "id", target_col] if col in df.columns]

    # Categorical columns from original script; filter to those present
    candidate_categorical = [
        "Gender",
        "Type of Travel",
        "Class",
        "Age Band",
        "Destination",
        "Continent",
    ]
    categorical_cols = [c for c in candidate_categorical if c in df.columns]

    # Numeric columns = all numbers excluding drop + categoricals
    numeric_cols = [
        c
        for c in df.columns
        if c not in drop_cols + categorical_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols, numeric_cols, target_col, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# -----------------------------
# Modeling
# -----------------------------

def build_log_reg_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )


def build_rf_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_estimators=100,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# -----------------------------
# Plotting Utilities
# -----------------------------

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_and_save_satisfaction_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Satisfied", data=df)
    plt.title("Passenger Satisfaction Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "satisfaction_distribution.png", dpi=150)
    plt.close()


def plot_and_save_age_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "Age" not in df.columns:
        return
    plt.figure(figsize=(12, 6))
    order = df["Age"].value_counts().index
    sns.countplot(x="Age", data=df, order=order)
    plt.title("Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png", dpi=150)
    plt.close()


def plot_and_save_corr_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    num_df = df.select_dtypes(include=["number"])  # excludes categorical strings
    if num_df.empty:
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_and_save_confusion_matrix(y_true, y_pred, labels, title, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues" if "Logistic" in title else "Greens",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_and_save_roc_curves(
    lr_fpr,
    lr_tpr,
    lr_auc,
    rf_fpr,
    rf_tpr,
    rf_auc,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.2f})")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_comparison.png", dpi=150)
    plt.close()


# -----------------------------
# Reporting Utilities
# -----------------------------

def save_text(text: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    try:
        # Available in sklearn >= 1.0
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: assemble names manually
        names: List[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if isinstance(cols, list):
                base_cols = cols
            else:
                # Column indices case (unlikely here)
                base_cols = [str(c) for c in cols]
            if name == "num":
                names.extend(base_cols)
            elif name == "cat":
                # If OneHotEncoder present, try categories_
                try:
                    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                    ohe_feature_names = ohe.get_feature_names_out(base_cols)
                    names.extend(list(ohe_feature_names))
                except Exception:
                    names.extend(base_cols)
            else:
                names.extend(base_cols)
        feature_names = names
    return feature_names


# -----------------------------
# Main
# -----------------------------

def execute(csv: str, output_dir: str = "outputs", test_size: float = 0.2, cv_folds: int = 5):
    output_dir = ensure_dir(output_dir)

    # Load data
    df = load_dataframe(csv)

    # Basic info outputs
    info_text = []
    info_text.append(f"Dataset Shape: {df.shape}")
    info_text.append("\nData Types and Non-Null Counts:\n")
    import io
    _buf = io.StringIO()
    df.info(buf=_buf)
    info_text.append(_buf.getvalue())
    info_text.append(f"\nTotal records in original dataset: {len(df)}")

    # Missing values summary
    missing = df.isnull().sum()
    info_text.append("\nMissing Values per Column (non-zero):\n")
    info_text.append(str(missing[missing > 0]))

    # Summary stats for numerical columns
    info_text.append("\nSummary Statistics (numeric):\n")
    info_text.append(str(df.describe(include=[np.number])))

    save_text("\n".join([str(x) for x in info_text]), output_dir / "dataset_info.txt")

    # EDA plots (saved only)
    if "Satisfied" in df.columns and df["Satisfied"].dtype == object:
        df["Satisfied"] = df["Satisfied"].map({"Y": 1, "N": 0})

    plot_and_save_satisfaction_distribution(df, output_dir)
    plot_and_save_age_distribution(df, output_dir)
    plot_and_save_corr_heatmap(df, output_dir)

    # Feature sets
    feature_cols, numeric_cols, target_col, categorical_cols = get_feature_sets(df)

    # Define X, y
    X = df[feature_cols]
    y = df[target_col]

    # Drop rows with missing target
    non_missing_mask = y.notna()
    X = X.loc[non_missing_mask]
    y = y.loc[non_missing_mask]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    # Preprocessor & Pipelines
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    lr_pipe = build_log_reg_pipeline(preprocessor)
    rf_pipe = build_rf_pipeline(preprocessor)

    # Fit and evaluate Logistic Regression
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    report_lr = classification_report(y_test, y_pred_lr)

    # Fit and evaluate Random Forest
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    # Save confusion matrices
    labels = ["Not Satisfied", "Satisfied"]
    plot_and_save_confusion_matrix(
        y_test, y_pred_lr, labels, "Logistic Regression - Confusion Matrix", output_dir / "confusion_matrix_logreg.png"
    )
    plot_and_save_confusion_matrix(
        y_test, y_pred_rf, labels, "Random Forest - Confusion Matrix", output_dir / "confusion_matrix_rf.png"
    )

    # Save ROC curves comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    plot_and_save_roc_curves(fpr_lr, tpr_lr, auc_lr, fpr_rf, tpr_rf, auc_rf, output_dir)

    # Cross-validation on full data for both models
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores_lr = cross_val_score(lr_pipe, X, y, cv=skf, scoring="accuracy")
    cv_scores_rf = cross_val_score(rf_pipe, X, y, cv=skf, scoring="accuracy")

    # Save reports
    metrics_text = []
    metrics_text.append("Model Performance (Test Set):")
    metrics_text.append(f"- Logistic Regression Accuracy: {acc_lr:.4f}")
    metrics_text.append(f"- Logistic Regression ROC AUC: {auc_lr:.4f}")
    metrics_text.append(f"- Random Forest Accuracy: {acc_rf:.4f}")
    metrics_text.append(f"- Random Forest ROC AUC: {auc_rf:.4f}")
    metrics_text.append("")
    metrics_text.append("Classification Report - Logistic Regression:\n" + report_lr)
    metrics_text.append("Classification Report - Random Forest:\n" + report_rf)
    metrics_text.append("")
    metrics_text.append(
        f"Logistic Regression {cv_folds}-fold CV Accuracy: {cv_scores_lr} (mean={cv_scores_lr.mean():.4f})"
    )
    metrics_text.append(
        f"Random Forest {cv_folds}-fold CV Accuracy: {cv_scores_rf} (mean={cv_scores_rf.mean():.4f})"
    )

    save_text("\n".join(metrics_text), output_dir / "metrics.txt")

    # Summary comparison table (CSV)
    comparison_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Test Accuracy": [acc_lr, acc_rf],
            "Mean CV Accuracy": [cv_scores_lr.mean(), cv_scores_rf.mean()],
            "Test ROC AUC": [auc_lr, auc_rf],
        }
    )
    comparison_df.to_csv(output_dir / "model_performance_comparison.csv", index=False)

    # Coefficients (LogReg) and Feature Importances (RF)
    feature_names = extract_feature_names_from_preprocessor(lr_pipe.named_steps["preprocess"])  # same preprocessor

    try:
        lr_coef = lr_pipe.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"Feature": feature_names[: len(lr_coef)], "Coefficient": lr_coef})
        coef_df = coef_df.sort_values(by="Coefficient", ascending=False)
        coef_df.to_csv(output_dir / "logreg_coefficients.csv", index=False)
    except Exception:
        pass

    try:
        rf_importances = rf_pipe.named_steps["clf"].feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names[: len(rf_importances)], "Importance": rf_importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)
        imp_df.to_csv(output_dir / "rf_feature_importances.csv", index=False)
    except Exception:
        pass

    # Also save a small README for outputs
    readme = (
        "This directory contains artifacts generated by the Silky Sky Airways modeling script:\n\n"
        "- dataset_info.txt: Dataset shape, dtypes, missing values, summary statistics\n"
        "- satisfaction_distribution.png: Histogram of target labels\n"
        "- age_distribution.png: Distribution of Ages (if present)\n"
        "- correlation_heatmap.png: Correlation heatmap of numerical features\n"
        "- confusion_matrix_logreg.png: Confusion matrix for Logistic Regression\n"
        "- confusion_matrix_rf.png: Confusion matrix for Random Forest\n"
        "- roc_curve_comparison.png: ROC curves and AUC for both models\n"
        "- metrics.txt: Text summary of metrics and classification reports\n"
        "- model_performance_comparison.csv: Side-by-side comparison table\n"
        "- logreg_coefficients.csv: Coefficients of Logistic Regression (if available)\n"
        "- rf_feature_importances.csv: Feature importances from Random Forest (if available)\n"
    )
    save_text(readme, output_dir / "README.txt")

    return output_dir


def run(csv: str, output_dir: str = "outputs", test_size: float = 0.2, cv_folds: int = 5):
    """Convenience entrypoint for notebooks (e.g., Google Colab)."""
    return execute(csv, output_dir, test_size, cv_folds)


def main() -> None:
    args = parse_args()
    execute(args.csv, args.output_dir, args.test_size, args.cv_folds)


if __name__ == "__main__":
    main()

