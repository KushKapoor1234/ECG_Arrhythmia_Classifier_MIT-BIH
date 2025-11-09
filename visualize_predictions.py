import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import wfdb

from src.feature_extractor import extract_features_for_beat, BEAT_CLASS_MAP
from src.signal_processor import pan_tompkins_detector

MODEL_PATH = "outputs/models/arrhythmia_classifier_pipeline.joblib"
DATA_DIR = "data"
OUT_DIR = "outputs/eval_outputs/visualizations"
RECORD = "100"
N_EXAMPLES = 6
WINDOW_MS = 150  # half window (ms)

os.makedirs(OUT_DIR, exist_ok=True)


def run(record=RECORD):
    model = joblib.load(MODEL_PATH)
    rec = wfdb.rdrecord(os.path.join(DATA_DIR, record))
    ann = wfdb.rdann(os.path.join(DATA_DIR, record), "atr")
    signal = rec.p_signal[:, 0]
    fs = int(rec.fs)
    samples = ann.sample
    symbols = ann.symbol

    valid_indices = [i for i, s in enumerate(symbols) if BEAT_CLASS_MAP.get(s) is not None]
    idxs = np.linspace(1, len(valid_indices) - 2, N_EXAMPLES, dtype=int)
    fig, axes = plt.subplots(N_EXAMPLES, 1, figsize=(10, 2 * N_EXAMPLES), sharex=False)

    for ax, j in zip(axes, idxs):
        sample = samples[valid_indices[j]]
        feat = extract_features_for_beat(signal, int(sample), samples[valid_indices], j, fs)
        if feat is None:
            continue
        X = np.array(feat).reshape(1, -1)
        pred = model.predict(X)[0]
        true_label = BEAT_CLASS_MAP[symbols[valid_indices[j]]]

        hsamp = int((WINDOW_MS / 1000.0) * fs)
        start = max(0, int(sample - hsamp))
        end = min(len(signal), int(sample + hsamp))
        t = np.arange(start, end) / fs
        ax.plot(t, signal[start:end], lw=0.8)
        ax.axvline(sample / fs, color="k", linestyle="--", alpha=0.6)
        ax.set_title(f"Beat @ {sample} | true={true_label} | pred={pred}")
        ax.set_ylabel("mV")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{record}_example_beats.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved visualization to {out_png}")


if __name__ == "__main__":
    run()
