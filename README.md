# ECG Arrhythmia Classifier (MIT-BIH)

A **reproducible, end‑to‑end biosignal pipeline** for ECG arrhythmia analysis integrating:

* classical **signal processing** (filtering, QRS detection)
* structured **beat‑level feature extraction**
* extended **time‑frequency analysis** (Lomb–Scargle, CWT scalograms, coherence)
* subject‑aware **ML evaluation** with Group‑CV
* full experiment tracking + generated artifacts

---

## TL;DR

* **Languages / Libraries:** Python, NumPy, SciPy, scikit‑learn, wfdb, pandas, matplotlib
* **Pipeline:** Load raw MIT‑BIH → filter → Pan‑Tompkins detector → beat windows → morphological + RR features → TF‑features (optional) → RandomForest classifier → Group‑CV
* **Artifacts generated:** confusion matrices, per‑fold metrics, per‑record TF CSVs, CWT scalograms, Lomb–Scargle plots, coherence maps, trained ML pipeline, metadata
* **Example accuracy:** ~91.7% CV accuracy, macro‑F1 ≈ 0.58

Now includes full **time‑frequency toolkit** integrated via:

* **Lomb–Scargle periodograms**
* **Cosinor rhythm regression** (optional)
* **Continuous Wavelet Transforms (Morlet)**
* **Short‑time coherence** between ECG leads

---

## Results Summary

| Component                    | Metric          | Mean       | StdDev | Median     |
| ---------------------------- | --------------- | ---------- | ------ | ---------- |
| **Classifier (5‑fold CV)**   | Accuracy        | **91.71%** | ±4.17% | —          |
|                              | Macro‑F1        | **0.5807** | ±0.038 | —          |
|                              | Macro‑Precision | **0.6643** | ±0.123 | —          |
| **Detector (Pan–Tompkins+)** | PPV             | **97.44%** | —      | **99.74%** |
|                              | Sensitivity     | **88.00%** | —      | **96.90%** |

---

## Updated Repository Structure

```
.
├── README.md
├── requirements.txt
├── run.py                          # orchestrates everything
├── src/
│   ├── data_loader.py              # loads MIT-BIH records
│   ├── signal_processor.py         # filtering + Pan‑Tompkins
│   ├── feature_extractor.py        # beat-level morphological + RR features
│   ├── features_tf.py              # NEW — TF feature wrappers
│   ├── tf_analysis.py              # NEW — Lomb-Scargle, CWT, coherence
│   ├── ml_trainer.py               # Group-aware CV + final model
│   └── visualizer.py               # plotting helpers
│
├── eval/
│   ├── infer_on_record.py
│   ├── analyze_detector.py
│   └── visualize_predictions.py
│
├── data/                           # MIT-BIH .dat/.hea/.atr (ignored)
└── outputs/
    ├── plots/                      # confusion matrices, detection plots
    ├── models/                     # saved ML pipeline
    └── tf/                         # NEW — per-record TF features + plots
        ├── record_100/
        │   ├── tf_features.csv
        │   ├── periodogram.png
        │   ├── scalogram.png
        │   ├── coherence.png
        │   └── tf_meta.json
        └── tf_features_by_record.csv
```

---

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Preparation

Download MIT‑BIH Arrhythmia Database from PhysioNet:
[https://physionet.org/content/mitdb/](https://physionet.org/content/mitdb/)

Place the `.dat`, `.hea`, `.atr` files into the `data/` directory:

```
data/
  100.dat
  100.hea
  100.atr
  101.dat
  ...
```
---

## Run Training + Evaluation

```bash
python run.py
```

Outputs saved under `outputs/`:

* `plots/confusion_matrix.csv`
* `plots/cv_fold_metrics.csv`
* `models/final_pipeline.joblib`
* `metadata.json`

Also generates **time‑frequency artifacts**:

* Lomb–Scargle periodograms
* CWT scalograms
* Lead‑pair coherence heatmaps
* Per‑record TF feature CSVs

---

## Evaluation & Inference

Pan–Tompkins detector analysis:

```bash
python eval/analyze_detector.py
```

Inference on a record:

```bash
python eval/infer_on_record.py
```

Beat visualization:

```bash
python eval/visualize_predictions.py
```

---

## Feature Definitions

Beat‑level features:

```
rr_pre_ms
rr_post_ms
qrs_amplitude_max_mv
qrs_amplitude_min_mv
qrs_area_mvs
qrs_width_ms
qrs_max_slope_mv_per_s
qrs_spectral_entropy
```

TF Features (per‑record):

```
cwt_bandpower_low
cwt_bandpower_mid
cwt_bandpower_high
cwt_total_power
lomb_peak_freq
lomb_peak_power
coh_low_mean
coh_mid_mean
coh_high_mean
```

---

## Reproducibility

* All parameters saved in `metadata.json`
* Pinned `requirements.txt`
* Deterministic seeds
* Group-aware CV prevents subject leakage
* TF artifacts automatically versioned in `outputs/tf/`

---

## Contact

**[kushkapoor.kk1234@gmail.com](mailto:kushkapoor.kk1234@gmail.com)**

---

## Data Citation

MIT-BIH Arrhythmia Database — Goldberger AL *et al.*, Circulation (2000).
