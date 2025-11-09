# ECG Arrhythmia Classifier (MIT-BIH)

**Reproducible signal-processing → ML pipeline** for beat-level arrhythmia classification from raw MIT-BIH ECG recordings (PhysioNet).

---

## TL;DR

* Languages / libs: **Python, NumPy, SciPy, scikit-learn, wfdb, imbalanced-learn, pandas, matplotlib**
* What it does: Loads raw MIT‑BIH records, applies robust filtering and QRS detection, extracts beat-level morphological & RR features, trains a RandomForest classifier with **5-fold StratifiedGroupKFold** (subject-wise folds), and exports a deployable pipeline plus evaluation artifacts.
* Example results (from my run): **91.71% mean accuracy**, **0.5807 macro‑F1**; detector **PPV 97.44%** (median 99.74%), **sensitivity 88.00%** (median 96.90%).

---

## 📊 Results Summary

| Component                    | Metric          | Mean       | StdDev | Median     |
| ---------------------------- | --------------- | ---------- | ------ | ---------- |
| **Classifier (5-Fold CV)**   | Accuracy        | **91.71%** | ±4.17% | —          |
|                              | Macro‑F1        | **0.5807** | ±0.038 | —          |
|                              | Macro‑Precision | **0.6643** | ±0.123 | —          |
| **Detector (Pan–Tompkins+)** | PPV             | **97.44%** | —      | **99.74%** |
|                              | Sensitivity     | **88.00%** | —      | **96.90%** |

These metrics are derived from `cv_metrics_summary.csv` (classifier) and the detector evaluation CSV generated in `results/eval_outputs/analysis/`.

---

## Repository structure

```
.
├── README.md                        # (this file)
├── requirements.txt                 # pinned python deps
├── run.py                           # training & evaluation orchestrator
├── src/                             # project source code
│   ├── data_loader.py               # loads local PhysioNet files
│   ├── signal_processor.py          # filtering + Pan-Tompkins detector
│   ├── feature_extractor.py         # beat-level feature extraction
│   ├── ml_trainer.py                # Group-CV, oversampling, model export
│   └── visualizer.py                # plotting helpers
├── eval/                            # evaluation & inference scripts
│   ├── infer_on_record.py
│   ├── infer_on_feature_csv.py
│   ├── analyze_detector.py
│   └── visualize_predictions.py
├── data/                            # **NOT** committed: MIT-BIH .dat/.hea/.atr files
└── outputs/                         # outputs (ignored by default; optional LFS for models)
    ├── plots/
    ├── models/
    └── eval_outputs/
```

---

## Quick setup

```bash
# create venv
python -m venv .venv
. .venv/bin/activate

# install packages
pip install -r requirements.txt
```

`requirements.txt` is pinned for reproducibility (numPy, SciPy, scikit-learn, wfdb, pandas, imbalanced-learn, matplotlib, joblib, tqdm).

---

## Prepare the dataset (MIT-BIH)

1. Download the MIT‑BIH Arrhythmia Database from PhysioNet: [https://physionet.org/content/mitdb/](https://physionet.org/content/mitdb/)
2. Place the downloaded files in `data/` so each record has `.dat`, `.hea`, `.atr` files, e.g. `data/100.dat`, `data/100.hea`, `data/100.atr`.

**Do not** commit raw dataset files to GitHub; keep `data/` in `.gitignore`.

---

## Run training & evaluation

Train the model and produce evaluation artifacts (confusion matrix, per-fold metrics, final pipeline):

```bash
python run.py
```

Outputs are saved under `results/`:

* `results/plots/confusion_matrix.png` — normalized (per-row) confusion matrix from out-of-fold predictions.
* `results/plots/cv_metrics_summary.csv` — per-fold accuracy/precision/recall/F1 and Mean/Std rows.
* `results/models/arrhythmia_classifier_pipeline.joblib` — compressed scikit-learn Pipeline (scaler + RandomForest).
* `results/metadata.json` — experiment config, filter params, feature names, seed, package versions.

---

## Evaluation & inference

There are helper scripts in `eval/`.

**Detector analysis** (Pan‑Tompkins performance):

```bash
python eval/analyze_detector.py
# writes results/eval_outputs/analysis/detector_summary_tol50ms.csv
```

**Inference on a single MIT-BIH record (uses annotation for ground truth)**:

```bash
python eval/infer_on_record.py
# writes results/eval_outputs/inference/<record>_predictions.csv and classification report
```

**Predict on a CSV of precomputed features**:

```bash
# Prepare data/features_to_predict.csv with columns exactly matching FEATURE_NAMES in src/feature_extractor.py
python eval/infer_on_feature_csv.py
# writes results/eval_outputs/inference/features_to_predict_preds.csv
```

**Visualize example beat windows with predictions**:

```bash
python eval/visualize_predictions.py
# writes results/eval_outputs/visualizations/<record>_example_beats.png
```

---

## Feature names & units

```text
rr_pre_ms               # milliseconds (ms)
rr_post_ms              # milliseconds (ms)
qrs_amplitude_max_mv    # millivolts (mV)
qrs_amplitude_min_mv    # millivolts (mV)
qrs_area_mvs            # mV * s
qrs_width_ms            # ms (approx. half-max width)
qrs_max_slope_mv_per_s  # mv/s
qrs_spectral_entropy    # unitless (bits)
```

Maintaining order & units is crucial when constructing feature CSVs for `infer_on_feature_csv.py` or when re-training.

---

## Reproducibility

* `metadata.json` records seed, filter params, feature names, package versions — include it alongside results for auditability.
* Use the pinned `requirements.txt` to recreate the same environment.
* The CV loop uses `StratifiedGroupKFold` (fallback to `GroupKFold` if not available) to prevent subject leakage.

---

## License

Include a `LICENSE` file (MIT recommended for academic code). Replace `YEAR` and `Your Name` as appropriate.

---

## Contact

Email id: — `kushkapoor.kk1234@gmail.com`

---

## Acknowledgements / Data citation

MIT-BIH Arrhythmia Database: Goldberger AL et al., PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation 2000;101(23):e215–e220.

---

*End of README*
