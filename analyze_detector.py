import os
import numpy as np
import pandas as pd
import wfdb
from src.signal_processor import pan_tompkins_detector

DATA_DIR = "data"
OUT_DIR = "outputs/eval_outputs/analysis"
os.makedirs(OUT_DIR, exist_ok=True)
TOL_MS = 50


def match_detections(detected, ann_samples, fs, tol_ms=TOL_MS):
    tol_samples = int((tol_ms / 1000.0) * fs)
    ann_matched = np.zeros(len(ann_samples), dtype=bool)
    det_matched = np.zeros(len(detected), dtype=bool)
    for i, a in enumerate(ann_samples):
        candidates = np.where(np.abs(detected - a) <= tol_samples)[0]
        if candidates.size > 0:
            ann_matched[i] = True
            det_matched[candidates[0]] = True
    TP = ann_matched.sum()
    FP = (~det_matched).sum()
    FN = (~ann_matched).sum()
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    return {"TP": int(TP), "FP": int(FP), "FN": int(FN), "sensitivity": float(sens), "ppv": float(ppv)}


def run_all():
    hea_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".hea")]
    rows = []
    for f in hea_files:
        rec = os.path.splitext(f)[0]
        try:
            r = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
            ann = wfdb.rdann(os.path.join(DATA_DIR, rec), "atr")
            fs = int(r.fs)
            detected = pan_tompkins_detector(r.p_signal[:, 0], fs)
            stats = match_detections(detected, ann.sample, fs)
            stats.update({"record": rec, "n_ann": len(ann.sample), "n_detected": len(detected)})
            rows.append(stats)
            print(f"{rec}: sens={stats['sensitivity']:.3f}, ppv={stats['ppv']:.3f}")
        except Exception as e:
            print(f"Skipping {rec} due to error: {e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"detector_summary_tol{TOL_MS}ms.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved detector summary to {out_csv}")


if __name__ == "__main__":
    run_all()
