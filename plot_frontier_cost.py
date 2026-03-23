"""
Frontier cost scatter: does reaching a longer time-horizon require more
inference spend?

X-axis: model's 50 % time horizon (from logistic fit of score ~ log2(minutes))
Y-axis: median AI-cost / human-cost for successful attempts within ±0.1 OOM
        of that frontier

Each point is one model, coloured by release date.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from _alt_common import (
    load_data, build_model_release, weighted_median_ci,
    TIME_X_TICKS, TIME_X_LABELS, COST_Y_TICKS, COST_Y_LABELS,
)

# ── load data ──────────────────────────────────────────────────────────

_, release_dates, _, headline, df = load_data()
MODELS = df["alias"].unique().tolist()
model_release, sorted_models = build_model_release(MODELS, release_dates)

# ── per-model: p50 from METR logistic fits + cost at frontier ────────

p50_map = dict(zip(headline["agent"], headline["p50"]))

results = []

for alias in sorted_models:
    adf = df[df["alias"] == alias].copy()
    if len(adf) < 20:
        continue

    # ── p50 from pre-computed logistic fits (invsqrt weighting, reg=0.1)
    p50_minutes = p50_map.get(alias)
    if p50_minutes is None or p50_minutes < 1 or p50_minutes > 5000:
        continue

    # ── cost at frontier: successes within ±0.1 OOM of p50 ──────────
    half_band = 0.1  # in log10 space = 1/10 of an order of magnitude
    log10_p50 = np.log10(p50_minutes)
    lo = 10 ** (log10_p50 - half_band)
    hi = 10 ** (log10_p50 + half_band)

    frontier_successes = adf[
        (adf["score_binarized"] == 1) &
        (adf["human_minutes"] >= lo) &
        (adf["human_minutes"] <= hi)
    ]

    # If too few successes near frontier, widen to ±0.2 OOM
    if len(frontier_successes) < 3:
        lo = 10 ** (log10_p50 - 0.2)
        hi = 10 ** (log10_p50 + 0.2)
        frontier_successes = adf[
            (adf["score_binarized"] == 1) &
            (adf["human_minutes"] >= lo) &
            (adf["human_minutes"] <= hi)
        ]

    if len(frontier_successes) < 5:
        print(f"  WARNING: {alias} has only {len(frontier_successes)} frontier "
              f"successes — too few for reliable median, skipping")
        continue

    ratios = frontier_successes["cost_ratio"].values
    weights = frontier_successes["invsqrt_task_weight"].values
    median_cost_ratio, ci_lo, ci_hi = weighted_median_ci(ratios, weights)

    short = alias.replace(" (Inspect)", "").strip()
    results.append({
        "alias": alias,
        "short": short,
        "p50_minutes": p50_minutes,
        "median_cost_ratio": median_cost_ratio,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "release_date": model_release[alias],
        "n_frontier": len(frontier_successes),
    })
    print(f"  {short:30s}  p50={p50_minutes:8.1f} min  "
          f"cost_ratio={median_cost_ratio:.4f}  "
          f"90% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
          f"({len(frontier_successes)} frontier successes in "
          f"[{lo:.1f}, {hi:.1f}] min)")

rdf = pd.DataFrame(results)

# ── plot ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 8))

# Colour by release date (continuous)
date_nums = np.array([d.toordinal() for d in rdf["release_date"]])
norm = mcolors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
cmap = mcm.plasma

# Error bars (bootstrap 90 % CI) – draw first so dots sit on top
for _, row in rdf.iterrows():
    color = cmap(norm(row["release_date"].toordinal()))
    ax.plot(
        [row["p50_minutes"], row["p50_minutes"]],
        [row["ci_lo"], row["ci_hi"]],
        color=color, linewidth=2, alpha=0.5, zorder=4,
        solid_capstyle="round",
    )
    # Small caps at ends
    cap_w = 0.04  # in log-space data coords
    for yval in [row["ci_lo"], row["ci_hi"]]:
        ax.plot(
            [row["p50_minutes"] * 10**(-cap_w), row["p50_minutes"] * 10**cap_w],
            [yval, yval],
            color=color, linewidth=2, alpha=0.5, zorder=4,
        )

sc = ax.scatter(
    rdf["p50_minutes"],
    rdf["median_cost_ratio"],
    c=date_nums, cmap=cmap, norm=norm,
    s=140, edgecolors="black", linewidths=0.8, zorder=5,
)

# Label each point
for _, row in rdf.iterrows():
    ax.annotate(
        row["short"],
        (row["p50_minutes"], row["median_cost_ratio"]),
        textcoords="offset points", xytext=(8, 6),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )

# Colourbar for release date – compact so it doesn't block labels
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.4, aspect=15, anchor=(0.0, 1.0))
cbar.set_label("Release Date", fontsize=9)
unique_dates = sorted(rdf["release_date"].unique())
# Just show first, middle, last
idx = [0, len(unique_dates) // 2, -1]
cbar_ticks = [unique_dates[i].toordinal() for i in idx]
cbar_labels = [pd.Timestamp(unique_dates[i]).strftime("%b %Y") for i in idx]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_labels, fontsize=8)

# Axes
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xticks(TIME_X_TICKS)
ax.set_xticklabels(TIME_X_LABELS, rotation=30, ha="right")
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

ax.set_yticks(COST_Y_TICKS)
ax.set_yticklabels(COST_Y_LABELS)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
           label="AI cost = human cost")

ax.set_xlabel("50% Time Horizon", fontsize=13)
ax.set_ylabel("AI Cost / Human Cost (at Frontier ±0.1 OOM)", fontsize=13)
ax.set_title(
    "METR Time Horizon Analysis: Cost at 50% Time Horizon",
    fontsize=14,
)

# Build legend with CI explanation
handles, labels = ax.get_legend_handles_labels()
ci_handle = plt.Line2D([0], [0], color="gray", linewidth=2, alpha=0.5)
handles.append(ci_handle)
labels.append("90% bootstrap CI\non median cost ratio")
ax.legend(handles, labels, loc="upper right", fontsize=10)

# Grid at every tick
ax.grid(False, which="both")
for yval in COST_Y_TICKS:
    ax.axhline(y=yval, color="grey", linewidth=0.5, alpha=0.3, zorder=0)
for xval in TIME_X_TICKS:
    ax.axvline(x=xval, color="grey", linewidth=0.5, alpha=0.3, zorder=0)

fig.tight_layout()
out_path = "frontier_cost.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nSaved to {out_path}")
