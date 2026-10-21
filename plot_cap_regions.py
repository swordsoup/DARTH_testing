#!/usr/bin/env python3
"""
plot_cap_regions.py

Shows how the capacity cap -- not the declared recall -- governs what recall a
DARTH run actually achieves.

The hypothesis: DARTH presents --target-recall as the knob, but the search can
never exceed its capacity cap (nprobe for IVF, efSearch for HNSW). The probe
loop in IndexIVF.cpp breaks at `ik < nprobe` whether or not the predictor ever
said stop. When the cap binds, the declared recall is unreachable and the cap,
not the predictor, decides the outcome.

Two layouts, chosen automatically:

  <= 3 caps   one scatter, the caps overlaid and colour-coded. Good for the
              three-region picture (tight / middling / generous).

  >  3 caps   small multiples: one scatter panel per cap, plus two summary
              curves -- mean recall vs cap, and how often the cap binds vs cap.
              Six overlaid clouds cannot be told apart by colour, so faceting is
              the honest form; the summary curves are where a sweep actually
              pays off, since they show the trend a 3-point comparison only
              hints at.

Pick a subset with --caps to force the overlaid view.

Usage
-----
    # full sweep -> small multiples + trend curves
    python3 plot_cap_regions.py --csv results/glove_cap_tr0.85.csv \
        --target 0.85 --cost dists

    # three caps overlaid, the mentor's three-region picture
    python3 plot_cap_regions.py --csv results/glove_cap_tr0.85.csv \
        --target 0.85 --cost dists --caps 20,100,1000

    # HNSW runs, where the cap column is efSearch
    python3 plot_cap_regions.py --csv ... --target 0.85 --cap-col efSearch
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless node: write files, don't open windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Palette
#
# First three slots of the validated categorical order. A scatter puts every
# pair of series on screen at once, so it needs all-pairs separation, and only
# the first three slots clear that gate (worst all-pairs CVD dE 9.2,
# normal-vision dE 24.0 on the light surface). A fourth slot would put yellow
# beside orange and fail -- hence the facet path for longer sweeps rather than
# inventing more hues.
#
# Aqua sits at 2.74:1 against the light surface, under the 3:1 bar, so the
# relief rule applies: every cap is also direct-labelled and carries its own
# marker shape. Identity is never colour-alone here.
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # blue, orange, aqua
MARKERS = ["o", "s", "^"]                       # secondary encoding
TARGET_LINE = "#52514e"                         # a reference, not a series

MAX_OVERLAY = len(SERIES)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(path, cap_col, cost_col, recall_col, caps_wanted):
    df = pd.read_csv(path)

    missing = [c for c in (cap_col, cost_col, recall_col) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path}: missing column(s) {missing}\n"
            f"columns present: {list(df.columns)}"
        )

    # One row per query per cap. DARTH's final-result logging already gives
    # that; --per-prediction-logging would not.
    if df.duplicated(subset=["qid", cap_col]).any():
        before = len(df)
        sort_cols = ["qid", cap_col] + (["step"] if "step" in df.columns else [])
        df = df.sort_values(sort_cols).groupby(["qid", cap_col], as_index=False).last()
        log(f"collapsed {before} rows to {len(df)} (one per query per cap)")

    if caps_wanted:
        have = set(df[cap_col].unique())
        miss = [c for c in caps_wanted if c not in have]
        if miss:
            raise SystemExit(
                f"--caps asked for {miss}, which are not in the CSV.\n"
                f"available: {sorted(have)}"
            )
        df = df[df[cap_col].isin(caps_wanted)]

    return df


def summarise(df, cap_col, cost_col, recall_col, target):
    """Per-cap table. The quantitative version of the plot."""
    rows = []
    for cap, g in df.groupby(cap_col):
        cap = int(cap)

        # `step` is the 0-based index of the last list visited, so a query that
        # exhausted its budget has step == cap - 1. A little slack for
        # off-by-one differences between the IVF and HNSW paths.
        capped = (g["step"] >= cap - 2).mean() if "step" in g.columns else np.nan
        calls = (g["r_predictor_calls"].mean()
                 if "r_predictor_calls" in g.columns else np.nan)

        rows.append({
            cap_col: cap,
            "n": len(g),
            "mean_recall": g[recall_col].mean(),
            "median_recall": g[recall_col].median(),
            "frac_below_target": (g[recall_col] < target).mean(),
            "mean_cost": g[cost_col].mean(),
            "median_cost": g[cost_col].median(),
            "frac_hit_cap": capped,
            "mean_predictor_calls": calls,
        })

    return pd.DataFrame(rows).sort_values(cap_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=INK_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def draw(shown, ax, color, marker):
    ax.scatter(shown[0], shown[1], s=9, alpha=0.28, color=color,
               marker=marker, linewidths=0, zorder=2)


def subsample(g, cost_col, recall_col, sample):
    s = g.sample(sample, random_state=0) if sample and len(g) > sample else g
    return s[cost_col].values, s[recall_col].values


# ---------------------------------------------------------------------------
# Layout A: <= 3 caps, overlaid
# ---------------------------------------------------------------------------

def plot_overlay(df, caps, cap_col, cost_col, recall_col, target, out,
                 logx, sample, summary):
    fig = plt.figure(figsize=(13.5, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.9, 1], hspace=0.30, wspace=0.20)

    ax = fig.add_subplot(gs[0, :])
    style_axes(ax)

    for i, cap in enumerate(caps):
        g = df[df[cap_col] == cap]
        draw(subsample(g, cost_col, recall_col, sample), ax, SERIES[i], MARKERS[i])

    # Centroids last so they sit above the cloud; the labels are the relief
    # required by the aqua contrast WARN.
    for i, cap in enumerate(caps):
        g = df[df[cap_col] == cap]
        mx, my = g[cost_col].mean(), g[recall_col].mean()
        ax.scatter(mx, my, s=190, color=SERIES[i], marker=MARKERS[i],
                   edgecolors=SURFACE, linewidths=2.0, zorder=6)

    ax.axhline(target, color=TARGET_LINE, ls=(0, (5, 3)), lw=1.6, zorder=3)
    ax.annotate(f"declared recall {target:g}",
                xy=(0.996, target), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", ha="right",
                fontsize=9.5, color=INK_2, style="italic")

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(cost_col, fontsize=10.5, color=INK_2)
    ax.set_ylabel(f"actual recall  ({recall_col})", fontsize=10.5, color=INK_2)
    ax.set_title(f"One declared recall, {len(caps)} capacity caps\n"
                 f"every run asked for recall {target:g}; only the cap differs",
                 fontsize=13, color=INK, weight="bold", loc="left", pad=14)
    ax.legend(handles=[
        Line2D([], [], marker=MARKERS[i], color="none",
               markerfacecolor=SERIES[i], markeredgecolor="none", markersize=9,
               label=(f"{cap_col} = {cap}"
                      f"   mean r = {df[df[cap_col] == cap][recall_col].mean():.3f}"
                      f"   (n={len(df[df[cap_col] == cap])})"))
        for i, cap in enumerate(caps)
    ], loc="lower right", frameon=True, facecolor=SURFACE,
       edgecolor=INK_MUTED, fontsize=9.5, labelcolor=INK_2)

    # recall distribution
    ax2 = fig.add_subplot(gs[1, 0])
    style_axes(ax2)
    for i, cap in enumerate(caps):
        g = df[df[cap_col] == cap]
        ax2.hist(g[recall_col], bins=40, density=True, histtype="step",
                 color=SERIES[i], linewidth=2.0, zorder=3)
        ax2.hist(g[recall_col], bins=40, density=True,
                 color=SERIES[i], alpha=0.16, zorder=2)
    ax2.axvline(target, color=TARGET_LINE, ls=(0, (5, 3)), lw=1.6, zorder=4)
    ax2.set_xlabel("actual recall", fontsize=10, color=INK_2)
    ax2.set_ylabel("density", fontsize=10, color=INK_2)
    ax2.set_title("Where recall actually lands", fontsize=11.5,
                  color=INK, weight="bold", loc="left", pad=8)

    # where the guarantee breaks down -- colour still follows the cap, the two
    # metrics separated by texture, so this reads against the scatter
    ax3 = fig.add_subplot(gs[1, 1])
    style_axes(ax3)
    x = np.arange(len(caps))
    w = 0.36
    for i, cap in enumerate(caps):
        row = summary[summary[cap_col] == cap].iloc[0]
        ax3.bar(x[i] - w / 2, row["frac_hit_cap"] * 100, w,
                color=SERIES[i], edgecolor=SURFACE, linewidth=2.0, zorder=3)
        ax3.bar(x[i] + w / 2, row["frac_below_target"] * 100, w,
                color=SERIES[i], edgecolor=SURFACE, linewidth=2.0,
                hatch="///", alpha=0.55, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(c) for c in caps])
    ax3.set_xlabel(cap_col, fontsize=10, color=INK_2)
    ax3.set_ylabel("% of queries", fontsize=10, color=INK_2)
    ax3.set_ylim(0, 105)
    ax3.set_title("Where the guarantee breaks down", fontsize=11.5,
                  color=INK, weight="bold", loc="left", pad=8)
    ax3.legend(handles=[
        Patch(facecolor=INK_MUTED, edgecolor=SURFACE, label="exhausted the cap"),
        Patch(facecolor=INK_MUTED, edgecolor=SURFACE, hatch="///", alpha=0.55,
              label="missed declared recall"),
    ], loc="upper right", frameon=True, facecolor=SURFACE,
       edgecolor=INK_MUTED, fontsize=9, labelcolor=INK_2)

    return fig


# ---------------------------------------------------------------------------
# Layout B: > 3 caps, small multiples + trend curves
# ---------------------------------------------------------------------------

def plot_facet(df, caps, cap_col, cost_col, recall_col, target, out,
               logx, sample, summary):
    ncols = min(3, len(caps))
    nrows = int(np.ceil(len(caps) / ncols))

    fig = plt.figure(figsize=(5.0 * ncols, 3.4 * nrows + 7.0), facecolor=SURFACE)
    gs = fig.add_gridspec(nrows + 2, ncols,
                          height_ratios=[1] * nrows + [1.25, 1.25],
                          hspace=0.45, wspace=0.18)

    # Each panel holds one series, so identity never rests on colour here --
    # the panel title carries it, and one hue throughout keeps the grid calm.
    xs_all = df[cost_col]
    ys_all = df[recall_col]
    xlim = (xs_all.min() * 0.95, xs_all.max() * 1.05)
    ylim = (max(0, ys_all.min() - 0.03), min(1.02, ys_all.max() + 0.03))

    for i, cap in enumerate(caps):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        style_axes(ax)
        g = df[df[cap_col] == cap]
        draw(subsample(g, cost_col, recall_col, sample), ax, SERIES[0], "o")

        my = g[recall_col].mean()
        ax.axhline(target, color=TARGET_LINE, ls=(0, (5, 3)), lw=1.4, zorder=3)
        ax.axhline(my, color=SERIES[1], lw=1.8, zorder=4)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"{cap_col} = {cap}      mean r = {my:.3f}",
                     fontsize=11, color=INK, weight="bold", loc="left", pad=6)
        if i % ncols == 0:
            ax.set_ylabel(f"recall ({recall_col})", fontsize=9.5, color=INK_2)
        else:
            ax.tick_params(labelleft=False)
        if i // ncols == nrows - 1:
            ax.set_xlabel(cost_col, fontsize=9.5, color=INK_2)
        else:
            ax.tick_params(labelbottom=False)

    fig.text(0.5, 0.995,
             f"Per-query recall vs cost, one panel per capacity cap\n"
             f"every run asked for declared recall {target:g}; only the cap differs",
             ha="center", va="top", fontsize=14, color=INK, weight="bold")
    fig.legend(handles=[
        Line2D([], [], color=TARGET_LINE, ls=(0, (5, 3)), lw=1.6,
               label=f"declared recall {target:g}"),
        Line2D([], [], color=SERIES[1], lw=2.0, label="achieved mean recall"),
    ], loc="upper right", bbox_to_anchor=(0.995, 0.993), frameon=True,
       facecolor=SURFACE, edgecolor=INK_MUTED, fontsize=9.5, labelcolor=INK_2)

    capvals = summary[cap_col].values

    # --- the money chart: does achieved recall track the cap? --------------
    ax1 = fig.add_subplot(gs[nrows, :])
    style_axes(ax1)
    ax1.plot(capvals, summary["mean_recall"], marker="o", ms=8, lw=2.0,
             color=SERIES[0], zorder=4, label="achieved mean recall")
    ax1.fill_between(capvals, summary["mean_recall"], target,
                     color=SERIES[0], alpha=0.10, zorder=2)
    ax1.axhline(target, color=TARGET_LINE, ls=(0, (5, 3)), lw=1.8, zorder=3)
    for cx, cy in zip(capvals, summary["mean_recall"]):
        ax1.annotate(f"{cy:.3f}", xy=(cx, cy), xytext=(0, 9),
                     textcoords="offset points", ha="center",
                     fontsize=9, color=INK)
    ax1.annotate(f"declared recall {target:g}", xy=(0.996, target),
                 xycoords=("axes fraction", "data"), xytext=(0, 7),
                 textcoords="offset points", ha="right", fontsize=9.5,
                 color=INK_2, style="italic")
    ax1.set_xscale("log")
    ax1.set_xticks(capvals)
    ax1.set_xticklabels([str(c) for c in capvals])
    ax1.minorticks_off()
    ax1.set_xlabel(cap_col, fontsize=10.5, color=INK_2)
    ax1.set_ylabel("mean recall", fontsize=10.5, color=INK_2)
    ax1.set_title("Achieved recall is set by the cap, not by the declared target",
                  fontsize=12, color=INK, weight="bold", loc="left", pad=8)

    # --- how often the cap binds -------------------------------------------
    ax2 = fig.add_subplot(gs[nrows + 1, :])
    style_axes(ax2)
    ax2.plot(capvals, summary["frac_hit_cap"] * 100, marker="o", ms=8, lw=2.0,
             color=SERIES[0], label="exhausted the cap")
    ax2.plot(capvals, summary["frac_below_target"] * 100, marker="s", ms=8,
             lw=2.0, color=SERIES[1], label="missed declared recall")
    ax2.set_xscale("log")
    ax2.set_xticks(capvals)
    ax2.set_xticklabels([str(c) for c in capvals])
    ax2.minorticks_off()
    ax2.set_ylim(-3, 105)
    ax2.set_xlabel(cap_col, fontsize=10.5, color=INK_2)
    ax2.set_ylabel("% of queries", fontsize=10.5, color=INK_2)
    ax2.set_title("Where the guarantee breaks down", fontsize=12,
                  color=INK, weight="bold", loc="left", pad=8)
    ax2.legend(loc="lower left", frameon=True, facecolor=SURFACE,
               edgecolor=INK_MUTED, fontsize=9.5, labelcolor=INK_2)
    # Two series, so both are also direct-labelled at the last point.
    for col, c, lbl in ((summary["frac_hit_cap"], SERIES[0], "exhausted cap"),
                        (summary["frac_below_target"], SERIES[1], "missed target")):
        ax2.annotate(lbl, xy=(capvals[-1], col.values[-1] * 100),
                     xytext=(-8, 10), textcoords="offset points", ha="right",
                     fontsize=9, color=c, weight="bold")

    return fig


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="joined CSV from build_recall_csv.py")
    ap.add_argument("--target", type=float, required=True,
                    help="the declared recall these runs were given")
    ap.add_argument("--out", help="output PNG (default: alongside the CSV)")
    ap.add_argument("--cap-col", default="nprobe",
                    help="column holding the capacity cap: nprobe for IVF, "
                         "efSearch for HNSW. default: nprobe")
    ap.add_argument("--caps",
                    help="comma-separated subset of caps to plot, e.g. "
                         "20,100,1000. Three or fewer get the overlaid view.")
    ap.add_argument("--cost", default="elaps_ms", dest="cost_col",
                    help="x-axis cost column. elaps_ms is wall clock; dists is "
                         "distance computations, which is hardware-independent "
                         "and usually better for a writeup. default: elaps_ms")
    ap.add_argument("--recall-col", default="r",
                    help="column holding actual recall. default: r")
    ap.add_argument("--logx", action="store_true",
                    help="log-scale the cost axis on the scatter panels; useful "
                         "when caps span an order of magnitude")
    ap.add_argument("--sample", type=int, default=4000,
                    help="max points drawn per cap, to keep the scatter legible. "
                         "0 draws everything. Summary stats always use all rows. "
                         "default: 4000")
    args = ap.parse_args()

    caps_wanted = None
    if args.caps:
        caps_wanted = [int(c) for c in args.caps.split(",") if c.strip()]

    out = args.out or os.path.splitext(args.csv)[0] + f"_caps_{args.cost_col}.png"

    df = load(args.csv, args.cap_col, args.cost_col, args.recall_col, caps_wanted)
    caps = sorted(int(c) for c in df[args.cap_col].unique())
    log(f"{len(df)} rows, caps: {caps}")

    summary = summarise(df, args.cap_col, args.cost_col, args.recall_col, args.target)

    if len(caps) <= MAX_OVERLAY:
        log(f"{len(caps)} caps -> overlaid scatter")
        fig = plot_overlay(df, caps, args.cap_col, args.cost_col, args.recall_col,
                           args.target, out, args.logx, args.sample, summary)
    else:
        log(f"{len(caps)} caps -> small multiples + trend curves "
            f"(use --caps to pick 3 for the overlaid view)")
        fig = plot_facet(df, caps, args.cap_col, args.cost_col, args.recall_col,
                         args.target, out, args.logx, args.sample, summary)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log(f"wrote {out}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 96)
    print(f"declared recall = {args.target}")
    print("=" * 96)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 96)

    csv_out = os.path.splitext(out)[0] + "_summary.csv"
    summary.to_csv(csv_out, index=False)
    log(f"wrote {csv_out}")

    print("\nread:")
    for _, row in summary.iterrows():
        cap = int(row[args.cap_col])
        gap = row["mean_recall"] - args.target
        verdict = ("meets declared recall" if gap >= -0.01
                   else f"MISSES declared recall by {abs(gap):.3f}")
        print(f"  {args.cap_col}={cap:>5}: mean recall {row['mean_recall']:.4f}  ->  {verdict}")
        if not np.isnan(row["frac_hit_cap"]):
            print(f"{'':>12}   {row['frac_hit_cap']:6.1%} of queries exhausted the cap"
                  f"   |   {row['mean_predictor_calls']:.1f} predictor calls/query")
    print("\nIf mean recall climbs toward the target as the cap grows -- while the")
    print("declared recall was identical throughout -- the cap is the operative")
    print("knob, and the declarative guarantee is conditional on it.")


if __name__ == "__main__":
    main()
