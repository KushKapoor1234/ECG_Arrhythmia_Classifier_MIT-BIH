import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Export filter params for metadata
FILTER_PARAMS = {"low": 5.0, "high": 40.0, "order": 3}


def bandpass_filter(sig, fs, low=FILTER_PARAMS["low"], high=FILTER_PARAMS["high"], order=FILTER_PARAMS["order"]):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")

    padlen = 3 * max(len(a), len(b))
    if len(sig) <= padlen:
        return sig
    return filtfilt(b, a, sig)


def pan_tompkins_detector(raw_signal, fs, tol_ms=50):
    """
    Improved Pan-Tompkins style detector with MAD-based adaptive threshold
    and simple post-filtering to reduce false positives.

    Returns array of detected R-peak sample indices.
    """
    # 1. Bandpass
    filtered_signal = bandpass_filter(raw_signal, fs)

    # 2. Derivative
    derivative_signal = np.diff(filtered_signal, prepend=filtered_signal[0])

    # 3. Squaring
    squared_signal = derivative_signal ** 2

    # 4. Moving window integration (150 ms)
    window_size = max(1, int(0.150 * fs))
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode="same")

    # 5. Adaptive threshold: median + k * MAD (robust to outliers)
    med = np.median(integrated_signal)
    mad = np.median(np.abs(integrated_signal - med))
    k = 1.25  # tuned factor; smaller -> more peaks (higher sensitivity), larger -> fewer FPs
    threshold = med + k * mad
    # minimal threshold floor (avoid zero)
    threshold = max(threshold, 1e-8)

    # 6. Find peaks above threshold with refractory period
    min_distance = max(1, int(0.2 * fs))  # 200 ms refractory
    # add a small prominence requirement relative to median
    prom = max(1e-6, 0.1 * np.max(integrated_signal - med))
    peaks, props = find_peaks(integrated_signal, height=threshold, distance=min_distance, prominence=prom)

    # 7. Post-filter: remove peaks that are too small in raw_signal window or too close
    # Compute raw signal peak amplitude in small window and keep only sufficiently large ones.
    cleaned = []
    for p in peaks:
        # window +/- 50ms
        w = max(1, int(0.05 * fs))
        start = max(0, p - w)
        end = min(len(raw_signal), p + w)
        local_max = np.max(np.abs(raw_signal[start:end]))
        # require local_max > median amplitude * factor (reduces FP)
        global_local_median = np.median(np.abs(raw_signal))
        if local_max >= 0.5 * global_local_median and local_max > 0:
            cleaned.append(p)
        else:
            # keep if integrated signal strongly exceeds threshold (prevents missing low amplitude true peaks)
            if integrated_signal[p] > (threshold * 1.5):
                cleaned.append(p)
    cleaned = np.array(cleaned, dtype=int)

    # final dedupe pass: ensure refractory spacing (keep the highest integrated peak in short windows)
    if cleaned.size == 0:
        return cleaned

    final = []
    last = -np.inf
    for p in cleaned:
        if p - last < min_distance:
            # choose the one with larger integrated value between last and p
            if len(final) == 0:
                final.append(p)
                last = p
            else:
                prev_p = final[-1]
                if integrated_signal[p] > integrated_signal[prev_p]:
                    final[-1] = p
                    last = p
        else:
            final.append(p)
            last = p

    return np.array(final, dtype=int)
