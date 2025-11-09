import numpy as np
from tqdm import tqdm
from scipy.signal import welch

# Define which annotation symbols are considered "beats"
BEAT_SYMBOLS = ["N", "L", "R", "V", "A", "a", "J", "S", "j", "e", "E", "/"]

# Grouped class mapping
BEAT_CLASS_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",  # Normal and Bundle Branch Blocks
    "V": "V",
    "E": "V",  # Ventricular Ectopic Beats
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",  # Supraventricular Ectopic Beats
    "/": "Q",  # Paced Beats
}

# Export feature names (extended)
FEATURE_NAMES = [
    "rr_pre_ms",
    "rr_post_ms",
    "qrs_amplitude_max_mv",
    "qrs_amplitude_min_mv",
    "qrs_area_mvs",
    "qrs_width_ms",
    "qrs_max_slope_mv_per_s",
    "qrs_spectral_entropy",
]

# helper: spectral entropy
def spectral_entropy(x, fs, nfft=None):
    if nfft is None:
        nfft = min(1024, len(x))
    # compute PSD (Welch), restrict to >0 Hz
    f, pxx = welch(x, fs=fs, nperseg=nfft)
    pxx = pxx.copy()
    pxx = pxx[f > 0]
    f = f[f > 0]
    if pxx.size == 0:
        return 0.0
    p = pxx / np.sum(pxx)
    # numerical stability
    p = np.maximum(p, 1e-12)
    se = -np.sum(p * np.log2(p))
    return float(se)


def estimate_qrs_width_ms(beat_window, fs):
    """
    Approximate QRS width using half-maximum width of absolute signal in window.
    Returns width in milliseconds or np.nan if cannot be measured.
    """
    x = np.abs(beat_window)
    if x.size < 3:
        return np.nan
    peak_idx = np.argmax(x)
    peak_val = x[peak_idx]
    if peak_val <= 0:
        return np.nan
    half = 0.5 * peak_val

    # find left crossing
    left = peak_idx
    while left > 0 and x[left] > half:
        left -= 1
    # find right crossing
    right = peak_idx
    while right < len(x) - 1 and x[right] > half:
        right += 1

    width_samples = max(1, right - left)
    width_ms = (width_samples / fs) * 1000.0
    return float(width_ms)


def build_feature_dataset(all_records):
    """
    Extract features for each valid beat, return X, y, groups.
    """
    X_all = []
    y_all = []
    groups_all = []

    print("Extracting features from records (enhanced features)...")
    for record in tqdm(all_records):
        signal = record["signal"][:, 0]
        fs = record["fs"]
        ann_samples = record["ann_samples"]
        ann_symbols = record["ann_symbols"]
        rec_name = record["name"]

        valid_indices = [i for i, sym in enumerate(ann_symbols) if BEAT_CLASS_MAP.get(sym) is not None]

        if not valid_indices:
            continue

        true_beat_samples = ann_samples[valid_indices]
        true_beat_labels = [BEAT_CLASS_MAP[ann_symbols[i]] for i in valid_indices]

        for i in range(len(true_beat_samples)):
            sample = int(true_beat_samples[i])
            label = true_beat_labels[i]

            # Need previous and next beat to compute RR; skip first/last
            if i == 0 or i == (len(true_beat_samples) - 1):
                continue

            features = extract_features_for_beat(signal, sample, true_beat_samples, i, fs)
            if features is None:
                continue

            X_all.append(features)
            y_all.append(label)
            groups_all.append(rec_name)

    return np.array(X_all, dtype=float), np.array(y_all, dtype=object), np.array(groups_all, dtype=object)


def extract_features_for_beat(signal, beat_sample, all_beat_samples, beat_index, fs):
    """
    Feature vector for one beat:
      - rr_pre_ms
      - rr_post_ms
      - qrs_amplitude_max_mv
      - qrs_amplitude_min_mv
      - qrs_area_mvs
      - qrs_width_ms
      - qrs_max_slope_mv_per_s
      - qrs_spectral_entropy
    """
    # RR in samples -> ms
    try:
        rr_pre_samples = int(beat_sample - all_beat_samples[beat_index - 1])
        rr_post_samples = int(all_beat_samples[beat_index + 1] - beat_sample)
    except Exception:
        return None

    rr_pre_ms = (rr_pre_samples / fs) * 1000.0
    rr_post_ms = (rr_post_samples / fs) * 1000.0

    # 100ms window (50ms before/after)
    window_size_samples = max(1, int(0.05 * fs))
    window_start = max(0, int(beat_sample - window_size_samples))
    window_end = min(len(signal), int(beat_sample + window_size_samples))

    beat_window = signal[window_start:window_end]

    # Guard small/empty windows
    if beat_window.size < max(3, int(0.01 * fs)):
        return None

    qrs_amplitude_max = float(np.max(beat_window))
    qrs_amplitude_min = float(np.min(beat_window))
    qrs_area = float(np.trapz(np.abs(beat_window))) / fs  # mV * seconds

    # width estimation (ms)
    qrs_width_ms = estimate_qrs_width_ms(beat_window, fs)
    if np.isnan(qrs_width_ms):
        qrs_width_ms = 0.0

    # max slope (mv / s) computed from first diff
    diffs = np.diff(beat_window)
    if diffs.size == 0:
        max_slope = 0.0
    else:
        # diffs per sample -> slope per second
        max_slope = float(np.max(np.abs(diffs)) * fs)

    # spectral entropy of window
    try:
        spec_ent = spectral_entropy(beat_window, fs)
    except Exception:
        spec_ent = 0.0

    return [
        rr_pre_ms,
        rr_post_ms,
        qrs_amplitude_max,
        qrs_amplitude_min,
        qrs_area,
        qrs_width_ms,
        max_slope,
        spec_ent,
    ]
