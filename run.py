import os
import json
import sys
import platform
import warnings
import numpy as np
import joblib

from src.data_loader import load_all_records, RECORDS_TO_EXCLUDE
from src.feature_extractor import build_feature_dataset, BEAT_CLASS_MAP, FEATURE_NAMES
from src.ml_trainer import train_and_evaluate
from src.signal_processor import pan_tompkins_detector, FILTER_PARAMS
from src.visualizer import plot_single_record

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Deterministic seed
SEED = 42
np.random.seed(SEED)

# --- Configuration ---
DATABASE_NAME = "data"  # Point to your local 'data' folder
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
EXAMPLE_RECORD = "100"  # A good record to visualize

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def save_metadata(metadata):
    """Saves the experiment metadata to a JSON file."""
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    try:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Experiment metadata saved to {path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")


def main():
    print("--- ECG Arrhythmia Classification Project ---")

    # --- 1. Load Data ---
    print(f"\nLoading all records from local folder '{DATABASE_NAME}'...")
    all_records = load_all_records(DATABASE_NAME)

    if not all_records:
        raise RuntimeError(
            f"No records loaded. Check '{DATABASE_NAME}' folder for .hea/.dat/.atr files."
        )
    print(f"Loaded {len(all_records)} valid records.")

    # --- 2. (Optional) Visualize one record to show SP ---
    print(f"\nProcessing and visualizing example record '{EXAMPLE_RECORD}'...")
    example = next((r for r in all_records if r["name"] == EXAMPLE_RECORD), None)

    if example is None:
        print(f"Warning: Example record {EXAMPLE_RECORD} not found. Skipping visualization.")
    else:
        signal = example["signal"][:, 0]  # Use first channel
        fs = example["fs"]
        r_peaks = pan_tompkins_detector(signal, fs)

        plot_path = os.path.join(PLOTS_DIR, f"{EXAMPLE_RECORD}_detection_plot.png")
        plot_single_record(
            signal,
            r_peaks,
            fs,
            title=f"Pan-Tompkins on Record {EXAMPLE_RECORD}",
            save_path=plot_path,
        )
        print(f"Saved example plot to {plot_path}")

    # --- 3. Build Feature Dataset (SP + ML Bridge) ---
    print("\nBuilding feature dataset from all records...")
    X, y, groups = build_feature_dataset(all_records)

    print(f"\nFeature extraction complete. Total beats: {len(y)}")
    print(f"Feature matrix shape: {X.shape} (Features: {FEATURE_NAMES})")
    print(f"Target vector shape: {y.shape}")
    print(f"Groups array shape: {groups.shape}")

    # --- 4. Train and Evaluate ML Model ---
    actual_class_names = sorted(list(set(BEAT_CLASS_MAP.values())))
    print(f"\nTraining and evaluating for classes: {actual_class_names}")

    (
        report,
        conf_matrix_path,
        metrics_csv_path,
        final_model_path,
        final_classes,
    ) = train_and_evaluate(
        X, y, groups, actual_class_names, MODELS_DIR, PLOTS_DIR, seed=SEED
    )

    print("\n--- Cross-Validation Classification Report ---")
    print(report)

    report_path = os.path.join(PLOTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- 5-Fold Stratified Group Cross-Validation Report ---\n\n")
        f.write(report)

    print(f"\nClassification report saved to {report_path}")
    print(f"Per-fold metrics saved to {metrics_csv_path}")
    print(f"Confusion matrix saved to {conf_matrix_path}")
    print(f"Final trained pipeline saved to {final_model_path}")

    # --- 5. Save Experiment Metadata ---
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "records_loaded": len(all_records),
        "records_excluded": RECORDS_TO_EXCLUDE,
        "total_beats_extracted": int(len(y)),
        "class_map": BEAT_CLASS_MAP,
        "final_classes": final_classes.tolist() if hasattr(final_classes, "tolist") else list(final_classes),
        "signal_processing": {"filter_type": "Butterworth Bandpass", "filter_params": FILTER_PARAMS},
        "features": {"names": FEATURE_NAMES, "count": len(FEATURE_NAMES)},
        "model": {
            "type": "RandomForestClassifier",
            "pipeline": ["StandardScaler", "RandomForestClassifier"],
            "n_estimators": 150,
            "class_weight": "balanced",
        },
        "validation": {"type": "StratifiedGroupKFold (fallback: GroupKFold)", "n_splits": 5},
        "seed": SEED,
        "env": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }

    # add package versions if available
    try:
        import numpy as _np
        import sklearn as _sk
        import wfdb as _wfdb
        import pandas as _pd

        metadata["env"].update(
            {
                "numpy": _np.__version__,
                "scikit-learn": _sk.__version__,
                "wfdb": _wfdb.__version__,
                "pandas": _pd.__version__,
            }
        )
    except Exception:
        pass

    save_metadata(metadata)

    print("\n--- Project Complete ---")


if __name__ == "__main__":
    main()
