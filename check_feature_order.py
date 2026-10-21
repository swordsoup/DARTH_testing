#!/usr/bin/env python3
"""
check_feature_order.py

Tests whether DARTH's C++ predictors feed features to LightGBM in the order the
model was trained on.

The model is called via LGBM_BoosterPredictForMatSingleRow, which takes a raw
double[] -- feature names are never checked, so a wrong order predicts happily
on scrambled input and returns plausible-looking garbage.

This replays the model twice over real training rows: once with the correct
column order, once with the order the C++ actually supplies. If the correct
order correlates well with true recall and the C++ order doesn't, the ordering
is the bug.

Run from the repo root:
    python3 check_feature_order.py
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

MODEL = ("predictor_models/ivf/darth/"
         "GLOVE100_nlist1000_nprobe100_s10000_k50_nestim100_lr0.1_li20_rl0_all_feats.txt")
DATA = ("training_data/ivf/training-data-generation/training/GLOVE100/50/"
        "nlist1000_nprobe100_qs10000_li20.txt")
LOGGING_INTERVAL = 20

# The order the model was trained on -- this is the `feature_names` line in the
# model file, which comes from all_feats in ivf_predictor_training.py.
TRAINED_ORDER = [
    "step", "dists", "inserts",
    "first_nn_dist", "nn_dist", "furthest_dist",
    "avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75",
]

# The order DARTHPredictorIVF::predictRecall actually builds its double[11].
# Read straight off the array initialiser in DeclarativeRecall.cpp.
# DARTHPredictorHNSW::predict_recall uses the identical order.
CPP_ORDER = [
    "step", "dists", "inserts",
    "first_nn_dist", "nn_dist",
    "avg_dist",        # <-- model expects furthest_dist here
    "furthest_dist",   # <-- model expects avg_dist here
    "variance",
    "percentile_50",   # <-- model expects percentile_25 here  (C++ calls it `median`)
    "percentile_25",   # <-- model expects percentile_50 here
    "percentile_75",
]


def main():
    booster = lgb.Booster(model_file=MODEL)

    model_names = booster.feature_name()
    print("model feature_names:")
    print("   ", " ".join(model_names))
    if model_names != TRAINED_ORDER:
        print("\n!! model's names differ from TRAINED_ORDER in this script.")
        print("   using the model's own names as truth.")
        trained = model_names
    else:
        trained = TRAINED_ORDER

    print("\nposition-by-position:")
    mismatches = 0
    for i, (want, got) in enumerate(zip(trained, CPP_ORDER)):
        flag = "" if want == got else "   <-- MISMATCH"
        if want != got:
            mismatches += 1
        print(f"  [{i:2d}] model={want:16s} cpp={got:16s}{flag}")
    print(f"\n{mismatches} of {len(trained)} positions disagree.")

    df = pd.read_csv(DATA, usecols=TRAINED_ORDER + ["r"], nrows=500000)
    # Same subsample the trainer used.
    d = df[df["dists"] % LOGGING_INTERVAL == 0]
    # Only rows where the heap was full -- predictRecall returns 0.0 early
    # otherwise, so those rows were never really predicted on.
    if "inserts" in d.columns:
        d = d[d["inserts"] >= 50]
    d = d.sample(n=min(5000, len(d)), random_state=0)
    print(f"\nevaluating on {len(d)} rows")

    actual = d["r"].values
    pred_correct = booster.predict(d[trained].values)
    pred_cpp = booster.predict(d[CPP_ORDER].values)

    print("\n" + "=" * 60)
    for label, pred in (("correct order", pred_correct),
                        ("C++ order    ", pred_cpp)):
        corr = np.corrcoef(pred, actual)[0, 1]
        mae = np.mean(np.abs(pred - actual))
        print(f"{label}   corr={corr:+.4f}   MAE={mae:.4f}   "
              f"mean_pred={pred.mean():.4f}")
    print("=" * 60)
    print(f"actual recall  mean={actual.mean():.4f}  std={actual.std():.4f}")

    print("\nfirst 15 rows:")
    print(pd.DataFrame({
        "actual": actual[:15],
        "correct": pred_correct[:15],
        "cpp": pred_cpp[:15],
    }).round(4).to_string(index=False))

    print("\ninterpretation:")
    print("  correct order correlates well + C++ order does not")
    print("    -> the C++ feature ordering is the bug.")
    print("  both correlate well")
    print("    -> ordering is fine, look elsewhere (feature *values* at")
    print("       search time may differ from what training logged).")
    print("  neither correlates")
    print("    -> the training data or the model itself is the problem.")


if __name__ == "__main__":
    sys.exit(main())
