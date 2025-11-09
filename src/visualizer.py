import matplotlib.pyplot as plt
import numpy as np


def plot_single_record(signal, r_peaks, fs, title, save_path):
    """
    Plots the first 10 seconds of a signal and its detected R-peaks.
    """
    plot_duration_samples = int(10 * fs)
    signal_slice = signal[:plot_duration_samples]
    time_axis = np.arange(len(signal_slice)) / fs

    peaks_in_slice = r_peaks[r_peaks < plot_duration_samples]

    plt.figure(figsize=(20, 8))
    plt.plot(time_axis, signal_slice, label="Raw ECG Signal (Channel 0)")
    if peaks_in_slice.size > 0:
        plt.plot(time_axis[peaks_in_slice], signal_slice[peaks_in_slice], "ro", label="Detected R-Peaks (Pan-Tompkins)")

    plt.title(title, fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude (mV)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
