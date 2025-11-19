# src/features_tf.py
"""
Minimal wrappers to extract compact TF features.

Provides:
 - extract_tf_features_from_signal(signal_array, fs)
 - extract_tf_features_from_beat_window(window, fs)
 - extract_coherence_summary(x, y, fs)
 - extract_tf_features_from_record(record, lead_index=0, lead2_index=None)
"""
import numpy as np
from typing import Dict, Any
from .tf_analysis import compute_lomb_scargle, compute_cwt, compute_short_time_coherence

def band_power_from_cwt(coeffs, freqs, band):
    fmin, fmax = band
    if coeffs.size == 0 or freqs.size == 0:
        return 0.0
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size == 0:
        return 0.0
    power = np.abs(coeffs[idx, :]) ** 2
    return float(power.mean())


def extract_tf_features_from_signal(signal_array: np.ndarray, fs: float) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if signal_array is None or len(signal_array) == 0:
        keys = ['cwt_bandpower_low', 'cwt_bandpower_mid', 'cwt_bandpower_high', 'cwt_total_power', 'lomb_peak_freq', 'lomb_peak_power']
        return {k: float('nan') for k in keys}
    coeffs, freqs = compute_cwt(signal_array, fs)
    bands = {'low': (0.5, 4.0), 'mid': (4.0, 15.0), 'high': (15.0, min(40.0, fs / 2.0))}
    for name, band in bands.items():
        feats[f'cwt_bandpower_{name}'] = band_power_from_cwt(coeffs, freqs, band)
    feats['cwt_total_power'] = float((np.abs(coeffs) ** 2).mean()) if coeffs.size else 0.0
    # Lomb on analytic envelope
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(signal_array))
        t = np.arange(len(signal_array)) / float(fs)
        freqs_ls, pgram = compute_lomb_scargle(t, env, fmax=2.0)
        if len(pgram):
            peak_idx = int(np.argmax(pgram))
            feats['lomb_peak_freq'] = float(freqs_ls[peak_idx])
            feats['lomb_peak_power'] = float(pgram[peak_idx])
        else:
            feats['lomb_peak_freq'] = 0.0
            feats['lomb_peak_power'] = 0.0
    except Exception:
        feats['lomb_peak_freq'] = float('nan'); feats['lomb_peak_power'] = float('nan')
    return feats


def extract_tf_features_from_beat_window(window: np.ndarray, fs: float) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if window is None or len(window) == 0:
        keys = ['beat_cwt_low', 'beat_cwt_mid', 'beat_cwt_high', 'beat_cwt_total', 'beat_lomb_peak_freq', 'beat_lomb_peak_power']
        return {k: float('nan') for k in keys}
    coeffs, freqs = compute_cwt(window, fs)
    bands = {'low': (0.5, 4.0), 'mid': (4.0, 15.0), 'high': (15.0, min(40.0, fs / 2.0))}
    for name, band in bands.items():
        feats[f'beat_cwt_{name}'] = band_power_from_cwt(coeffs, freqs, band)
    feats['beat_cwt_total'] = float((np.abs(coeffs) ** 2).mean()) if coeffs.size else 0.0
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(window))
        t = np.arange(len(window)) / float(fs)
        freqs_ls, pgram = compute_lomb_scargle(t, env, fmax=5.0)
        if len(pgram):
            peak_idx = int(np.argmax(pgram))
            feats['beat_lomb_peak_freq'] = float(freqs_ls[peak_idx])
            feats['beat_lomb_peak_power'] = float(pgram[peak_idx])
        else:
            feats['beat_lomb_peak_freq'] = 0.0
            feats['beat_lomb_peak_power'] = 0.0
    except Exception:
        feats['beat_lomb_peak_freq'] = float('nan'); feats['beat_lomb_peak_power'] = float('nan')
    return feats


def extract_coherence_summary(x, y, fs):
    if x is None or y is None:
        return {'coh_low_mean': float('nan'), 'coh_mid_mean': float('nan'), 'coh_high_mean': float('nan')}
    try:
        times, freqs, coh = compute_short_time_coherence(x, y, fs)
        if coh.size == 0 or freqs.size == 0:
            return {'coh_low_mean': 0.0, 'coh_mid_mean': 0.0, 'coh_high_mean': 0.0}
        def band_mean(band):
            idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
            if idx.size == 0:
                return 0.0
            return float(coh[:, idx].mean())
        return {
            'coh_low_mean': band_mean((0.5, 4.0)),
            'coh_mid_mean': band_mean((4.0, 15.0)),
            'coh_high_mean': band_mean((15.0, min(40.0, fs / 2.0)))
        }
    except Exception:
        return {'coh_low_mean': float('nan'), 'coh_mid_mean': float('nan'), 'coh_high_mean': float('nan')}


def extract_tf_features_from_record(record: dict, lead_index: int = 0, lead2_index: int = None) -> Dict[str, Any]:
    """
    Convenience wrapper that extracts per-record TF summary features and optional coherence.
    Expects record dict to contain:
      - 'name' (str)
      - 'signal' (1D or 2D array-like)
      - 'fs' (sampling freq)
    Returns flat dict with 'record' key + tf features.
    """
    out: Dict[str, Any] = {}
    rec_name = record.get('name', 'unknown')
    out['record'] = rec_name
    sig = record.get('signal', None)
    fs = record.get('fs', None)
    if sig is None or fs is None:
        # fill NaNs for expected keys
        keys = ['cwt_bandpower_low','cwt_bandpower_mid','cwt_bandpower_high','cwt_total_power','lomb_peak_freq','lomb_peak_power',
                'coh_low_mean','coh_mid_mean','coh_high_mean']
        for k in keys:
            out[k] = float('nan')
        return out

    arr = np.asarray(sig)
    # normalize to (channels, samples)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        # it's likely (samples, channels)
        arr = arr.T

    nchan = arr.shape[0]
    lead0_idx = lead_index if lead_index < nchan else 0
    lead0 = arr[lead0_idx, :]

    # per-record signal TF features
    feats = extract_tf_features_from_signal(lead0, float(fs))
    out.update(feats)

    # coherence summary if second lead provided
    if lead2_index is not None and lead2_index < nchan:
        lead1 = arr[lead2_index, :]
        coh = extract_coherence_summary(lead0, lead1, float(fs))
        out.update(coh)
    else:
        out.update({'coh_low_mean': float('nan'), 'coh_mid_mean': float('nan'), 'coh_high_mean': float('nan')})

    return out
