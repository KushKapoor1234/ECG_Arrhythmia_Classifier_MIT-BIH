import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# StratifiedGroupKFold may not be available in older sklearn; fallback gracefully
try:
    from sklearn.model_selection import StratifiedGroupKFold

    SGKF_AVAILABLE = True
except Exception:
    from sklearn.model_selection import GroupKFold as StratifiedGroupKFold

    SGKF_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

# new import for balancing
from imblearn.over_sampling import RandomOverSampler


def train_and_evaluate(X, y, groups, class_names, models_dir, plots_dir, seed=42):
    """
    Performs a 5-Fold StratifiedGroup (fallback Group) cross-validation.
    Applies RandomOverSampler on the training fold to mitigate class imbalance.
    Returns (report_text, confusion_matrix_path, metrics_csv_path, final_model_path, final_classes)
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    n_splits = 5

    # Basic validity checks
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty feature matrix or labels passed to train_and_evaluate.")

    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < n_splits:
        raise ValueError(
            f"Number of unique groups ({unique_groups.shape[0]}) < n_splits ({n_splits}). Reduce n_splits."
        )

    # Create the splitter
    if SGKF_AVAILABLE:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        print("Warning: StratifiedGroupKFold not available, using GroupKFold fallback (no stratification).")
        sgkf = StratifiedGroupKFold(n_splits=n_splits)

    all_y_test = []
    all_y_pred = []
    fold_metrics_list = []

    print(f"Starting {n_splits}-Fold {'StratifiedGroupKFold' if SGKF_AVAILABLE else 'GroupKFold'} Cross-Validation...")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Print class distributions
        print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        print("Train dist (pre-oversample):", dict(Counter(y_train)))
        print("Test dist:", dict(Counter(y_test)))

        # --- Oversample the minority classes on TRAIN set only ---
        ros = RandomOverSampler(random_state=seed)
        try:
            X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
            print("Train dist (post-oversample):", dict(Counter(y_train_res)))
        except Exception as e:
            print("Oversampling failed, continuing without it:", e)
            X_train_res, y_train_res = X_train, y_train

        # Pipeline (scaler + classifier) -- reinstantiate per fold
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1
                    ),
                ),
            ]
        )

        # Fit on the oversampled training set
        pipeline.fit(X_train_res, y_train_res)
        y_pred = pipeline.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        # per-fold macro metrics
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        fold_metrics_list.append(
            {
                "fold": fold + 1,
                "accuracy": float(acc),
                "precision_macro": float(p),
                "recall_macro": float(r),
                "f1_macro": float(f1),
                "n_test_samples": int(len(y_test)),
            }
        )

    print("\nCross-Validation complete.")

    # Classification report aggregated
    report = classification_report(all_y_test, all_y_pred, target_names=class_names, labels=class_names, zero_division=0)

    # Save per-fold metrics to CSV (numeric-only mean/std)
    metrics_df = pd.DataFrame(fold_metrics_list)
    num_cols = metrics_df.select_dtypes(include=[np.number]).columns
    mean_row = metrics_df[num_cols].mean().to_dict()
    std_row = metrics_df[num_cols].std().to_dict()
    mean_row["fold"] = "Mean"
    std_row["fold"] = "StdDev"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    metrics_csv_path = os.path.join(plots_dir, "cv_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, float_format="%.4f")

    report += "\n\nMean Metrics (from folds):\n"
    report += metrics_df[metrics_df["fold"] == "Mean"].to_string(index=False)
    report += "\n\nStdDev Metrics (from folds):\n"
    report += metrics_df[metrics_df["fold"] == "StdDev"].to_string(index=False)

    # Train final pipeline on entire dataset (optionally oversampled)
    print("\nTraining final model on all data (with oversampling)...")
    ros_final = RandomOverSampler(random_state=seed)
    try:
        X_res_full, y_res_full = ros_final.fit_resample(X, y)
        print("Full train dist (post-oversample):", dict(Counter(y_res_full)))
    except Exception as e:
        print("Full-dataset oversampling failed, training on raw dataset:", e)
        X_res_full, y_res_full = X, y

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)),
        ]
    )
    final_pipeline.fit(X_res_full, y_res_full)

    # Save final pipeline compressed
    final_model_path = os.path.join(models_dir, "arrhythmia_classifier_pipeline.joblib")
    joblib.dump(final_pipeline, final_model_path, compress=3)

    final_classes = None
    try:
        final_classes = final_pipeline.named_steps["clf"].classes_
    except Exception:
        final_classes = np.unique(y)

    # Confusion matrix (from all folds)
    plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(all_y_test, all_y_pred, ax=ax, cmap="Blues", normalize="true", xticks_rotation="vertical", labels=class_names)
    plt.title(f"Normalized Confusion Matrix (from {n_splits}-Fold CV)")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return report, plot_path, metrics_csv_path, final_model_path, final_classes
