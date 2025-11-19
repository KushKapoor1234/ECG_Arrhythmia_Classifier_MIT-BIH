# src/feature_extractor.py
"""
Build beat-level feature dataset, integrating TF features per-beat.

Primary function:
 - build_feature_dataset(all_records, pre_ms=200, post_ms=400, lead=0)

Returns:
 - X: numpy array (n_beats, n_features)
 - y: numpy array (n_beats,) integer labels (unlabeled beats -> -1)
 - groups: numpy array (n_beats,) group id (record name for subject-wise CV)
Also writes per-record 'tf_beats.csv' into out_tf_root/record_<name>/
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .signal_processor import pan_tompkins_detector
from .features_tf import extract_tf_features_from_beat_window, extract_coherence_summary

# conservative default label map (adjust if your annotations use other symbols)
BEAT_CLASS_MAP = {
    'N': 0,  # normal
    'L': 0,  # left bundle branch block (map to normal or adjust)
    'R': 0,  # right bundle branch block
    'V': 1,  # ventricular ectopic
    'A': 2,  # atrial ectopic
    'F': 3,  # fusion
    'Q': 4,  # unknown / paced / other
}

# base (non-TF) beat features:
BASE_FEATURE_NAMES = [
    'win_mean', 'win_std', 'win_max', 'win_min', 'win_ptp', 'win_energy', 'pre_rr', 'post_rr'
]

# TF beat feature names (prefix 'beat_')
BEAT_TF_NAMES = [
    'beat_cwt_low', 'beat_cwt_mid', 'beat_cwt_high', 'beat_cwt_total',
    'beat_lomb_peak_freq', 'beat_lomb_peak_power'
]
COHERENCE_NAMES = ['coh_low_mean', 'coh_mid_mean', 'coh_high_mean']

FEATURE_NAMES = BASE_FEATURE_NAMES + BEAT_TF_NAMES + COHERENCE_NAMES


def _ensure_sig_shape(sig):
    """Return signal as (n_channels, n_samples)."""
    arr = np.asarray(sig)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        # if shape (n_samples, n_channels) convert to (channels, samples)
        if arr.shape[0] > arr.shape[1]:
            return arr.T
        return arr
    raise ValueError("Unsupported signal shape")


def _extract_base_features(window):
    """
    Simple time-domain features for a window (1D array).
    """
    w = np.asarray(window, dtype=float)
    feats = {
        'win_mean': float(np.mean(w)),
        'win_std': float(np.std(w)),
        'win_max': float(np.max(w)),
        'win_min': float(np.min(w)),
        'win_ptp': float(np.ptp(w)),
        'win_energy': float(np.sum(w.astype(float) ** 2))
    }
    return feats


def build_feature_dataset(all_records, pre_ms=200, post_ms=400, lead=0, lead2=None, out_tf_root="outputs/tf"):
    """
    Build beat-level dataset:
      - all_records: list of dicts with keys: 'name','signal','fs' and optionally 'ann_locs' and 'ann_labels'
      - pre_ms, post_ms: window around R-peak
      - lead: which channel index to use for beat windows
      - lead2: optional second lead index for coherence features
    Returns X, y, groups (y entries are ints; unlabeled beats -> -1)
    Also writes per-record 'tf_beats.csv' into out_tf_root/record_<name>/tf_beats.csv
    """
    os.makedirs(out_tf_root, exist_ok=True)
    rows = []
    rec_counter = 0

    for rec in tqdm(all_records, desc="records"):
        rec_name = rec.get('name', f"rec{rec_counter}")
        fs = float(rec.get('fs', 360.0))
        sig = rec.get('signal', None)
        if sig is None:
            rec_counter += 1
            continue
        sig = _ensure_sig_shape(sig)  # (channels, samples)
        nchan, nsamp = sig.shape
        lead_idx = min(lead, nchan - 1)
        lead2_idx = None if lead2 is None else min(lead2, nchan - 1)

        # attempt to use annotations if present
        ann_locs = None
        ann_labels = None
        # support various key names used by different loaders
        for k in ('ann_locs', 'ann_samples', 'r_peaks', 'r_locations'):
            if k in rec:
                ann_locs = np.asarray(rec[k]).astype(int)
                break
        for k in ('ann_labels', 'ann_symbols', 'labels', 'symbols'):
            if k in rec:
                ann_labels = np.asarray(rec[k])
                break

        # if no ann_locs then detect peaks
        if ann_locs is None or len(ann_locs) == 0:
            try:
                r_peaks = pan_tompkins_detector(sig[lead_idx, :], fs)
                ann_locs = np.asarray(r_peaks).astype(int)
                ann_labels = None
            except Exception:
                ann_locs = np.array([], dtype=int)
                ann_labels = None

        if ann_locs is None or len(ann_locs) == 0:
            rec_counter += 1
            continue

        pre_samps = int(round(pre_ms * fs / 1000.0))
        post_samps = int(round(post_ms * fs / 1000.0))

        # per-record TF beat CSV
        rec_out_dir = os.path.join(out_tf_root, f"record_{rec_name}")
        os.makedirs(rec_out_dir, exist_ok=True)
        beat_rows = []

        # compute RR intervals for pre_rr/post_rr where possible
        rlocs = ann_locs
        rr_intervals = np.diff(rlocs) / fs  # seconds between r-peaks, length n-1

        for i, r in enumerate(rlocs):
            s = int(r - pre_samps)
            e = int(r + post_samps)
            if s < 0 or e > nsamp:
                continue
            window = sig[lead_idx, s:e]

            # base time-domain features
            base_feats = _extract_base_features(window)

            # rr features
            pre_rr = rr_intervals[i-1] if i-1 >= 0 and i-1 < len(rr_intervals) else np.nan
            post_rr = rr_intervals[i] if i < len(rr_intervals) else np.nan
            base_feats['pre_rr'] = float(pre_rr) if not np.isnan(pre_rr) else float('nan')
            base_feats['post_rr'] = float(post_rr) if not np.isnan(post_rr) else float('nan')

            # per-beat TF features
            tf_feats = extract_tf_features_from_beat_window(window, fs)

            # coherence features if second lead present
            coh_feats = {}
            if lead2_idx is not None and lead2_idx < nchan:
                win2 = sig[lead2_idx, s:e]
                coh_feats = extract_coherence_summary(window, win2, fs)
            else:
                coh_feats = {n: float('nan') for n in COHERENCE_NAMES}

            # build label: prefer ann_labels if present
            label = None
            if ann_labels is not None and i < len(ann_labels):
                lab = ann_labels[i]
                # if symbol-based mapping
                if isinstance(lab, str) and lab in BEAT_CLASS_MAP:
                    label = BEAT_CLASS_MAP[lab]
                else:
                    try:
                        label = int(lab)
                    except Exception:
                        label = None

            # --- NEW: Do not drop unlabeled beats; mark them ---
            if label is None:
                labelled_flag = False
                label_val = -1
            else:
                labelled_flag = True
                label_val = int(label)

            # combine all features
            combined = {'record': rec_name, 'beat_index': int(i), 'peak_sample': int(r)}
            combined.update(base_feats)
            combined.update(tf_feats)
            combined.update(coh_feats)
            combined['label'] = label_val
            combined['is_labelled'] = bool(labelled_flag)

            # append to per-record and global lists
            beat_rows.append(combined)
            rows.append(combined)

        # write per-record per-beat tf CSV (may include unlabeled beats with label=-1)
        if beat_rows:
            df_rec = pd.DataFrame(beat_rows)
            try:
                df_rec.to_csv(os.path.join(rec_out_dir, "tf_beats.csv"), index=False)
            except Exception:
                pass

        rec_counter += 1

    # finalize dataframe
    if len(rows) == 0:
        # return empty numpy arrays but keep FEATURE_NAMES length
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=int), np.zeros((0,), dtype=object)

    df_all = pd.DataFrame(rows)

    # ensure FEATURE_NAMES present in dataframe; fill missing with nan
    for c in FEATURE_NAMES:
        if c not in df_all.columns:
            df_all[c] = float('nan')

    # finalize X,y,groups
    X = df_all[FEATURE_NAMES].values.astype(float)
    y = df_all['label'].fillna(-1).astype(int).values
    groups = np.array(df_all['record'].values)
    return X, y, groups
