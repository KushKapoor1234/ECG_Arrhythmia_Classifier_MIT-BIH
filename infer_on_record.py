import os
import joblib
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from src.feature_extractor import extract_features_for_beat, FEATURE_NAMES, BEAT_CLASS_MAP
from src.signal_processor import pan_tompkins_detector

# --- CONFIG ---
MODEL_PATH = "outputs/models/arrhythmia_classifier_pipeline.joblib"
DATA_DIR = "data"
OUT_DIR = "outputs/eval_outputs/inference"
RECORD = "100"
CHANNEL_IDX = 0

os.makedirs(OUT_DIR, exist_ok=True)


def load_pipeline(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def extract_features_and_labels_for_record(record_name):
    rec_path = os.path.join(DATA_DIR, record_name)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")

    signal = record.p_signal[:, CHANNEL_IDX]
    fs = int(record.fs)

    ann_samples = ann.sample
    ann_symbols = ann.symbol

    valid_indices = [i for i, s in enumerate(ann_symbols) if BEAT_CLASS_MAP.get(s) is not None]
    if len(valid_indices) <= 2:
        raise RuntimeError("Not enough valid annotated beats for feature extraction.")

    samples = ann_samples[valid_indices]
    labels = [BEAT_CLASS_MAP[ann_symbols[i]] for i in valid_indices]

    X, y, beat_positions = [], [], []
    for j in range(1, len(samples) - 1):
        sample = int(samples[j])
        feat = extract_features_for_beat(signal, sample, samples, j, fs)
        if feat is None:
            continue
        X.append(feat)
        y.append(labels[j])
        beat_positions.append(sample)

    return np.array(X, dtype=float), np.array(y, dtype=object), fs, beat_positions


def run(record=RECORD):
    model = load_pipeline(MODEL_PATH)
    X, y_true, fs, beat_samples = extract_features_and_labels_for_record(record)
    if X.size == 0:
        print("No features extracted. Exiting.")
        return

    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("Classification report (inference on record):\n", report_text)

    cm = confusion_matrix(y_true, y_pred, labels=model.named_steps["clf"].classes_)
    print("Confusion matrix:\n", cm)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["beat_sample"] = beat_samples[: len(df)]
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    if proba is not None:
        classes = model.named_steps["clf"].classes_
        for i, c in enumerate(classes):
            df[f"prob_{c}"] = proba[:, i]

    out_csv = os.path.join(OUT_DIR, f"{record}_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved per-beat predictions to {out_csv}")

    with open(os.path.join(OUT_DIR, f"{record}_classification_report.txt"), "w") as f:
        f.write(report_text)

    summary = {
        "record": record,
        "n_beats": len(y_true),
        "class_distribution_true": dict(Counter(y_true)),
        "class_distribution_pred": dict(Counter(y_pred)),
    }
    pd.Series(summary).to_csv(os.path.join(OUT_DIR, f"{record}_summary.csv"))
    print("Saved summary and report.")


if __name__ == "__main__":
    run()
