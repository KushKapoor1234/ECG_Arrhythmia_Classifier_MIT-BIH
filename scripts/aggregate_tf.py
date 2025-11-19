#!/usr/bin/env python3
"""
Aggregate per-record per-beat TF CSVs into a single master CSV.

Usage:
python scripts/aggregate_tf.py --tf_root outputs/tf --out combined_tf_beats.csv
"""

import os
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tf_root', default='outputs/tf')
    p.add_argument('--out', default='outputs/tf/combined_tf_beats.csv')
    args = p.parse_args()

    rows = []
    for d in sorted(os.listdir(args.tf_root)):
        full = os.path.join(args.tf_root, d)
        if not os.path.isdir(full):
            continue
        csvp = os.path.join(full, "tf_beats.csv")
        if os.path.exists(csvp):
            try:
                df = pd.read_csv(csvp)
                rows.append(df)
            except Exception as e:
                print(f"Failed to read {csvp}: {e}")
    if not rows:
        print("No per-record beat TF CSVs found.")
        return
    df_all = pd.concat(rows, ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_all.to_csv(args.out, index=False)
    print("Wrote aggregated TF beats to:", args.out)

if __name__ == "__main__":
    main()
