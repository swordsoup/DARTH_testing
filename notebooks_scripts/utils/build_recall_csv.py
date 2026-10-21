#!/usr/bin/env python3
"""
build_recall_csv.py

Joins DARTH per-query search output against GloVe token metadata to produce a
single analysis-ready CSV: one row per (run, query), carrying recall, search
cost, DARTH's hardness features, and the token's word and frequency rank.

Input
-----
One or more DARTH --output files. Those are already CSV, with a `qid` column
that indexes into the query set, and `r` as the final recall for that query.

The join key is positional: qid N corresponds to row N of the .fvecs query
file, entry N of <queryset>.words.json, and entry N of <queryset>.freqrank.json.
This holds because build_glove_twitter.py wrote all three in the same order.

Run parameters (nprobe, k, query set) are not recorded in the DARTH output
file, so they are taken from the filename or passed explicitly. Every row
carries them, which is what lets you concatenate several runs into one frame
and compare across them.

Usage
-----
    # single run, params parsed from filename
    python3 build_recall_csv.py \
        --inputs /tmp/glove_nprobe10.txt \
        --out /proj/.../results/glove_recall.csv

    # several runs at once
    python3 build_recall_csv.py \
        --inputs /tmp/glove_nprobe*.txt \
        --out /proj/.../results/glove_recall_sweep.csv

    # explicit params, and a different query set
    python3 build_recall_csv.py \
        --inputs /tmp/glove_english.txt \
        --queryset query.english.1K \
        --nprobe 50 --k 50 \
        --out /proj/.../results/glove_english.csv
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd


PROCESSED_DIR = "/mnt/data/darth-datasets/processed/GLOVE100"


# ---------------------------------------------------------------------------
# Column documentation
#
# DARTH emits these per query. Keeping the meanings here because the header in
# the raw output is terse and easy to misread six weeks from now.
# ---------------------------------------------------------------------------

COLUMN_NOTES = {
    "qid": "query index, 0-based, indexes into the query set",
    "step": "search step at which this row was logged",
    "dists": "distance computations performed",
    "elaps_ms": "elapsed search time, milliseconds",
    "inserts": "insertions into the result heap",
    "first_nn_dist": "distance to the true nearest neighbour (from groundtruth)",
    "nn_dist": "distance to the nearest neighbour found so far",
    "avg_dist": "mean distance across the current candidate set",
    "furthest_dist": "largest distance in the current candidate set",
    "variance": "variance of candidate distances",
    "std": "standard deviation of candidate distances",
    "range": "furthest minus nearest candidate distance",
    "skewness": "skew of the candidate distance distribution",
    "kurtosis": "kurtosis of the candidate distance distribution",
    "energy": "sum of squared candidate distances",
    "nn10_dist": "distance to the 10th nearest neighbour found",
    "nn_to_first": "nn_dist / first_nn_dist, ratio-based hardness proxy",
    "nn10_to_first": "nn10_dist / first_nn_dist",
    "RDE": "DARTH recall distance estimate",
    "TDR": "DARTH target distance ratio",
    "NRS": "DARTH normalised rank sum",
    "feats_collect_time_ms": "time spent computing these features",
    "r": "RECALL for this query -- the dependent variable",
    "word": "the GloVe token this query vector represents",
    "freqrank": "line number in the frequency-sorted GloVe file; 0 = most common",
    "queryset": "which query file this run used",
    "nprobe": "IVF lists probed",
    "k": "neighbours requested",
    "run": "source filename",
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_params_from_name(path):
    """Pull nprobe / nlist / k out of a filename if they are encoded there.

    Matches the naming the DARTH experiment scripts use (nprobe100, k50) as
    well as the simpler names from a manual run (glove_nprobe10.txt).
    Returns whatever it finds; missing values come back as None.
    """
    name = os.path.basename(path)
    found = {}
    for key, pattern in (
        ("nlist", r"nlist(\d+)"),
        ("nprobe", r"nprobe(\d+)"),
        ("k", r"(?:^|[_.])k(\d+)"),
    ):
        m = re.search(pattern, name)
        if m:
            found[key] = int(m.group(1))
    return found


def load_metadata(queryset):
    """Load the word and frequency-rank sidecars for a query set."""
    words_path = os.path.join(PROCESSED_DIR, f"{queryset}.words.json")
    freq_path = os.path.join(PROCESSED_DIR, f"{queryset}.freqrank.json")

    for p in (words_path, freq_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing metadata file: {p}\n"
                f"expected sidecars for query set '{queryset}'. "
                f"available in {PROCESSED_DIR}:\n  "
                + "\n  ".join(sorted(
                    f for f in os.listdir(PROCESSED_DIR)
                    if f.endswith(".words.json")
                ))
            )

    with open(words_path) as f:
        words = json.load(f)
    with open(freq_path) as f:
        freq = json.load(f)

    if len(words) != len(freq):
        raise SystemExit(
            f"metadata length mismatch for {queryset}: "
            f"{len(words)} words vs {len(freq)} ranks"
        )
    return words, freq


def load_run(path, queryset, words, freq, override):
    """Read one DARTH output file and attach metadata."""
    df = pd.read_csv(path)

    if "qid" not in df.columns:
        raise SystemExit(f"{path}: no 'qid' column -- is this a DARTH --output file?")
    # plain-ivf writes the recall column as `r`; darth mode writes `r_actual`
    # (alongside `r_predicted`). Normalise to `r` so downstream is uniform.
    if "r" not in df.columns:
        for alt in ("r_actual", "actual_recall", "Ract"):
            if alt in df.columns:
                df = df.rename(columns={alt: "r"})
                break
        else:
            raise SystemExit(
                f"{path}: no recall column (looked for r, r_actual, "
                f"actual_recall, Ract). columns present: {list(df.columns)}")

    # DARTH writes one row per query in plain-ivf mode, but emits a row per
    # prediction checkpoint under --per-prediction-logging. Keep the last row
    # per query either way, which is the final state of the search.
    n_rows = len(df)
    if df["qid"].duplicated().any():
        df = df.sort_values(["qid", "step"]).groupby("qid", as_index=False).last()
        log(f"  {os.path.basename(path)}: collapsed {n_rows} rows "
            f"to {len(df)} (per-prediction logging detected)")

    # Bounds check before indexing into the metadata lists.
    max_qid = int(df["qid"].max())
    if max_qid >= len(words):
        raise SystemExit(
            f"{path}: qid {max_qid} exceeds query set '{queryset}' "
            f"({len(words)} entries). Wrong --queryset?"
        )

    df["word"] = df["qid"].map(lambda i: words[int(i)])
    df["freqrank"] = df["qid"].map(lambda i: freq[int(i)])
    df["queryset"] = queryset
    df["run"] = os.path.basename(path)

    params = parse_params_from_name(path)
    params.update({k: v for k, v in override.items() if v is not None})
    for key in ("nlist", "nprobe", "k"):
        df[key] = params.get(key, np.nan)

    return df


def summarise(df):
    """Print a quick read on the joined frame."""
    log("")
    log("=" * 68)

    for run, g in df.groupby("run", sort=False):
        npr = g["nprobe"].iloc[0]
        npr = "?" if pd.isna(npr) else int(npr)
        log(f"\n{run}  (nprobe={npr}, n={len(g)})")
        log(f"  recall  mean={g['r'].mean():.4f}  median={g['r'].median():.4f}  "
            f"min={g['r'].min():.4f}  max={g['r'].max():.4f}")

        at_ceiling = (g["r"] >= 0.999).mean()
        log(f"  at ceiling (r>=0.999): {at_ceiling:5.1%}")
        if at_ceiling > 0.30:
            log(f"    ^ heavy ceiling effect -- correlations here are attenuated.")
            log(f"      lower nprobe to open up the range.")

        # Only report correlations over columns that are actually present and
        # have some variance to speak of.
        log("  correlation with recall:")
        for c in ("freqrank", "first_nn_dist", "nn_to_first", "nn10_to_first",
                  "dists", "variance", "std", "RDE", "NRS"):
            if c in g.columns and g[c].nunique() > 1:
                log(f"    {c:16s} {g['r'].corr(g[c]):+.3f}")

    log("")
    log("=" * 68)


def main():
    global PROCESSED_DIR

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="DARTH --output files to join")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--queryset", default="query.10K",
                    help="query set name, matches the .words.json prefix "
                         "(default: query.10K)")
    ap.add_argument("--processed-dir", default=PROCESSED_DIR,
                    help="directory holding the .words.json / .freqrank.json sidecars")
    ap.add_argument("--nlist", type=int, help="override nlist for all runs")
    ap.add_argument("--nprobe", type=int, help="override nprobe for all runs")
    ap.add_argument("--k", type=int, help="override k for all runs")
    ap.add_argument("--notes", action="store_true",
                    help="write a companion .columns.txt describing each column")
    args = ap.parse_args()

    PROCESSED_DIR = args.processed_dir

    words, freq = load_metadata(args.queryset)
    log(f"query set '{args.queryset}': {len(words)} tokens")

    override = {"nlist": args.nlist, "nprobe": args.nprobe, "k": args.k}

    frames = []
    for path in args.inputs:
        if not os.path.exists(path):
            log(f"  skipping missing file: {path}")
            continue
        frames.append(load_run(path, args.queryset, words, freq, override))

    if not frames:
        raise SystemExit("no input files could be read")

    df = pd.concat(frames, ignore_index=True)

    # Put the columns you actually care about at the front. Everything else
    # keeps its original order behind them.
    lead = ["run", "queryset", "nlist", "nprobe", "k", "qid", "word", "freqrank", "r"]
    lead = [c for c in lead if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    summarise(df)
    log(f"\nwrote {len(df)} rows to {args.out}")

    if args.notes:
        notes_path = os.path.splitext(args.out)[0] + ".columns.txt"
        with open(notes_path, "w") as f:
            for c in df.columns:
                f.write(f"{c:24s} {COLUMN_NOTES.get(c, '')}\n")
        log(f"wrote column notes to {notes_path}")


if __name__ == "__main__":
    main()
