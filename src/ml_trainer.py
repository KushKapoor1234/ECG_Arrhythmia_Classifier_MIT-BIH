# src/ml_trainer.py
"""
Minimal ML training & evaluation helper for ECG project.

Function:
  train_and_evaluate(X, y, groups, class_names, models_dir, plots_dir, seed=42, n_splits=5)

Behavior:
 - Automatically adjusts n_splits if there are fewer unique groups than requested.
 - Tries to use StratifiedGroupKFold; falls back to GroupKFold or StratifiedKFold as needed.
 - Returns (classification_report_text, conf_matrix_path, metrics_csv_path, final_model_path, classes_array)
"""
import os
import warnings
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
import joblib

# Try import of StratifiedGroupKFold (newer sklearn versions)
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    HAS_SGKF = True
except Exception:
    StratifiedGroupKFold = None  # type: ignore
    HAS_SGKF = False


def _ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: List[Any],
    models_dir: str,
    plots_dir: str,
    seed: int = 42,
    n_splits: int = 5,
) -> Tuple[str, str, str, str, np.ndarray]:
    """
    Train and evaluate a RandomForest pipeline with group-aware CV.
    Returns:
      report_text, confusion_matrix_path, metrics_csv_path, final_model_path, classes_array
    """
    _ensure_dir(models_dir)
    _ensure_dir(plots_dir)

    # Basic validation
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if X.size == 0 or y.size == 0:
        raise ValueError("Empty feature matrix or labels passed to train_and_evaluate.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if groups.shape[0] != y.shape[0]:
        # try broadcasting groups if single group provided
        if groups.size == 1:
            groups = np.repeat(groups, y.shape[0])
        else:
            raise ValueError("groups must have same length as y or be a single value.")

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Adjust n_splits to number of groups when necessary
    if n_groups < 2:
        warnings.warn("Only one unique group found. Falling back to sample-based StratifiedKFold.")
        # use stratified kfold on samples
        cv = StratifiedKFold(n_splits=min(n_splits, max(2, min(5, len(y)))), shuffle=True, random_state=seed)
    else:
        # ensure requested splits isn't greater than groups
        if n_splits > n_groups:
            old = n_splits
            n_splits = n_groups
            print(f"[ml_trainer] Warning: requested n_splits={old} > unique groups={n_groups}; reducing n_splits -> {n_splits}")

        # prefer StratifiedGroupKFold if available
        if HAS_SGKF and StratifiedGroupKFold is not None:
            try:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            except Exception:
                # fallback to GroupKFold
                cv = GroupKFold(n_splits=n_splits)
        else:
            # fallback to GroupKFold (no stratification across classes)
            cv = GroupKFold(n_splits=n_splits)

    # Build a simple pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]
    )

    y_true_all = []
    y_pred_all = []
    fold_metrics = []

    # Manual CV loop because GroupKFold/StratifiedGroupKFold accept groups in split()
    fold_idx = 0
    for train_idx, test_idx in cv.split(X, y, groups):
        fold_idx += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Fit
        pipeline.fit(Xtr, ytr)
        ypred = pipeline.predict(Xte)

        # Collect
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(ypred.tolist())

        # Per-fold metrics
        acc = float(accuracy_score(yte, ypred))
        f1 = float(f1_score(yte, ypred, average="weighted", zero_division=0))
        prec = float(precision_score(yte, ypred, average="weighted", zero_division=0))
        rec = float(recall_score(yte, ypred, average="weighted", zero_division=0))
        fold_metrics.append({"fold": fold_idx, "acc": acc, "f1_weighted": f1, "precision_weighted": prec, "recall_weighted": rec, "n_test": len(yte)})

    # Overall report
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # If no predictions produced (shouldn't happen), raise
    if y_pred_all.size == 0:
        raise RuntimeError("Cross-validation produced no predictions (empty).")

    # Classification report
    try:
        target_names = [str(c) for c in class_names]
        report_text = classification_report(y_true_all, y_pred_all, target_names=target_names, zero_division=0)
    except Exception:
        # fallback without target names
        report_text = classification_report(y_true_all, y_pred_all, zero_division=0)

    # Confusion matrix (aggregate)
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_path = os.path.join(plots_dir, "confusion_matrix.csv")
    # save as CSV for easy viewing
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(cm_path, index=False)
    # Save per-fold metrics
    metrics_csv_path = os.path.join(plots_dir, "cv_fold_metrics.csv")
    pd.DataFrame(fold_metrics).to_csv(metrics_csv_path, index=False)

    # Train final model on all data
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]
    )
    final_pipeline.fit(X, y)

    final_model_path = os.path.join(models_dir, "final_pipeline.joblib")
    joblib.dump(final_pipeline, final_model_path)

    # Save textual report
    report_path = os.path.join(plots_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Return values (report_text, cm_path, metrics_csv_path, final_model_path, classes_array)
    classes_array = np.array(sorted(list(set(y.tolist()))))
    return report_text, cm_path, metrics_csv_path, final_model_path, classes_array
