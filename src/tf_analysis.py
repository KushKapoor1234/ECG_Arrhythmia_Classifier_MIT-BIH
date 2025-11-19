# src/tf_analysis.py
"""
Time-frequency analysis utilities (minimal & robust).

Small optimization: number of CWT scales can be lowered via env var CWT_SCALES
to speed up compute during debugging.
"""
import os
import json
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import coherence


def compute_lomb_scargle(times: np.ndarray,
                         values: np.ndarray,
                         freqs: np.ndarray = None,
                         fmax: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 2 or values.size < 2 or np.all(np.isclose(values, values[0])):
        if freqs is None:
            freqs = np.linspace(0.001, fmax, 200)
        return freqs, np.zeros_like(freqs)
    values = values - np.nanmean(values)
    if freqs is None:
        freqs = np.linspace(0.001, fmax, 2000)
    ang = 2.0 * np.pi * freqs
    pgram = signal.lombscargle(times, values, ang, precenter=False, normalize=True)
    return freqs, pgram


def fit_cosinor(times: np.ndarray,
                values: np.ndarray,
                period: float) -> Dict[str, Any]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 3:
        return {"mesor": float("nan"), "amplitude": float("nan"), "acrophase": float("nan"), "r2": float("nan"), "beta": [float("nan")] * 3}
    omega = 2.0 * np.pi / float(period)
    X = np.column_stack([np.cos(omega * times), np.sin(omega * times), np.ones_like(times)])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    a, b, c = float(beta[0]), float(beta[1]), float(beta[2])
    amplitude = float(np.hypot(a, b))
    acrophase = float(np.arctan2(-b, a))
    yhat = X.dot(beta)
    ssres = float(np.sum((values - yhat) ** 2))
    sstot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - ssres / (sstot + 1e-12)
    return {"mesor": c, "amplitude": amplitude, "acrophase": acrophase, "r2": r2, "beta": [float(b) for b in beta]}


def compute_cwt(signal_in: np.ndarray,
                fs: float,
                widths: np.ndarray = None,
                wavelet_w: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal CWT wrapper. Respect env var CWT_SCALES to reduce work (debugging).
    """
    x = np.asarray(signal_in, dtype=float)
    if x.size == 0:
        return np.empty((0, 0)), np.empty((0,))

    try:
        from scipy.signal import cwt, morlet2
        if widths is None:
            # allow user to override number of scales for speed
            try:
                n_scales = int(os.environ.get("CWT_SCALES", "48"))
                if n_scales < 4:
                    n_scales = 4
            except Exception:
                n_scales = 48
            freqs = np.linspace(1.0, min(50.0, fs / 2.0), n_scales)
            widths = (wavelet_w * fs) / (2.0 * np.pi * freqs)
        else:
            widths = np.asarray(widths, dtype=float)
            freqs = (wavelet_w * fs) / (2.0 * np.pi * widths)
        def wfn(M, s): return morlet2(M, s, w=wavelet_w)
        coeffs = cwt(x, wfn, widths)
        return coeffs, freqs
    except Exception:
        # fallback to pywt if scipy cwt not available
        try:
            import pywt
            wavelet = 'cmor1.5-1.0'
            if widths is None:
                try:
                    n_scales = int(os.environ.get("CWT_SCALES", "48"))
                    if n_scales < 4:
                        n_scales = 4
                except Exception:
                    n_scales = 48
                freqs = np.linspace(1.0, min(50.0, fs / 2.0), n_scales)
                center = pywt.central_frequency(wavelet)
                scales = center * fs / freqs
            else:
                scales = np.asarray(widths, dtype=float)
                center = pywt.central_frequency(wavelet)
                freqs = pywt.scale2frequency(wavelet, scales) * fs
            coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
            return coeffs, freqs
        except Exception as e:
            raise ImportError("CWT requires scipy.signal.cwt or PyWavelets (pywt). Install pywt or upgrade SciPy.") from e


def compute_short_time_coherence(x: np.ndarray,
                                 y: np.ndarray,
                                 fs: float,
                                 window_sec: float = 2.0,
                                 step_sec: float = 0.5,
                                 nperseg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.array([]), np.array([]), np.array([[]])
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[:n]; y = y[:n]
    wlen = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if wlen < 4:
        raise ValueError("window_sec too small for sampling rate")
    if nperseg is None:
        nperseg = min(256, wlen)
    nwin = max(1, (len(x) - wlen) // step + 1)
    times = []
    coh_list = []
    freqs = None
    for i in range(nwin):
        s = i * step
        e = s + wlen
        xs = x[s:e]; ys = y[s:e]
        f, Cxy = coherence(xs, ys, fs=fs, nperseg=nperseg)
        if freqs is None:
            freqs = f
        coh_list.append(Cxy)
        times.append((s + e) / 2.0 / fs)
    if not coh_list:
        return np.array([]), np.array([]), np.array([[]])
    coh_matrix = np.vstack(coh_list)
    return np.array(times), freqs, coh_matrix


# --- plotting helpers ---
def _ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_periodogram(freqs: np.ndarray, power: np.ndarray, outpath: str, title: str = "Lomb-Scargle"):
    _ensure_dir_for_file(outpath)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    plt.plot(freqs, power)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_scalogram(coeffs: np.ndarray, freqs: np.ndarray, fs: float, outpath: str, vmin=None, vmax=None, cmap='magma'):
    _ensure_dir_for_file(outpath)
    import matplotlib.pyplot as plt
    if coeffs.size == 0:
        plt.figure(figsize=(4, 2)); plt.text(0.5, 0.5, "no data", ha="center"); plt.axis('off'); plt.savefig(outpath); plt.close(); return
    T = coeffs.shape[1] / fs
    times = np.linspace(0, T, coeffs.shape[1])
    power = np.abs(coeffs) ** 2
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(times, freqs, power, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Scalogram (CWT power)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_coherence(times: np.ndarray, freqs: np.ndarray, coh_mat: np.ndarray, outpath: str, vmin=0.0, vmax=1.0, cmap='viridis'):
    _ensure_dir_for_file(outpath)
    import matplotlib.pyplot as plt
    if coh_mat.size == 0:
        plt.figure(figsize=(4, 2)); plt.text(0.5, 0.5, "no coherence data", ha="center"); plt.axis('off'); plt.savefig(outpath); plt.close(); return
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(times, freqs, coh_mat.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Short-time coherence")
    plt.colorbar(label='Coherence')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_json(obj: dict, path: str):
    _ensure_dir_for_file(path)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
