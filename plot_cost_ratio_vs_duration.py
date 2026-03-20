"""
Plot AI-cost / human-cost (efficiency) vs task time-horizon for selected METR Time Horizon Analysis models.

X-axis: task time horizon (human_minutes) – log-scaled
Y-axis: AI inference cost / human cost – log-scaled

Each model's tasks are grouped into log-spaced bins by human_minutes.
For each bin the median cost ratio is plotted, with a shaded IQR band.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from _alt_common import (
    load_data, get_color,
    TIME_X_TICKS, TIME_X_LABELS, COST_Y_TICKS, COST_Y_LABELS,
)


def _weighted_quantile(values, weights, q):
    """Weighted quantile. q in [0,1]."""
    idx = np.argsort(values)
    sv, sw = values[idx], weights[idx]
    cum = np.cumsum(sw)
    cum = (cum - 0.5 * sw) / cum[-1]  # midpoint interpolation
    return np.interp(q, cum, sv)


# ── load data ──────────────────────────────────────────────────────────

df, release_dates, _, _, _ = load_data()

# ── models to plot ─────────────────────────────────────────────────────

MODELS = [
    "Claude 3.5 Sonnet (New) (Inspect)",
    "o1 (Inspect)",
    "o3 (Inspect)",
    "Claude 4.1 Opus (Inspect)",
    "Claude Opus 4.5 (Inspect)",
    "GPT-5.2",
    "Gemini 3 Pro",
]

df = df[df["alias"].isin(MODELS)]

# Keep only successful attempts
df = df[df["score_binarized"] == 1]

# ── release-date ordering ─────────────────────────────────────────────

aliases = sorted(
    MODELS,
    key=lambda a: str(release_dates.get(a, "9999-12-31")),
)

color_map = {a: get_color(a) for a in aliases}
color_map.update({
    "Claude 4.1 Opus (Inspect)": "#d62728",
    "Claude Opus 4.5 (Inspect)": "#2ca02c",
    "GPT-5.2": "#1a1a1a",
})
markers = ["o", "s", "D", "^", "v", "P"]
marker_map = {a: markers[i % len(markers)] for i, a in enumerate(aliases)}

# ── bin tasks by human_minutes ─────────────────────────────────────────

# Create ~15 log-spaced bins covering the range of human_minutes
min_minutes = df["human_minutes"].min()
max_minutes = df["human_minutes"].max()
bin_edges = np.logspace(np.log10(max(min_minutes, 1.5)), np.log10(min(max_minutes, 2000)), 18)
bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

# ── plot ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(13, 8))

for alias in aliases:
    adf = df[df["alias"] == alias].copy()
    if len(adf) < 10:
        continue

    adf["bin"] = pd.cut(adf["human_minutes"], bins=bin_edges, labels=bin_centres)

    # Weighted median and IQR using invsqrt_task_weight
    xs, y_meds, y_los, y_his = [], [], [], []
    for bc, grp in adf.groupby("bin", observed=True):
        if len(grp) < 5:
            continue
        vals = grp["cost_ratio"].values
        wts = grp["invsqrt_task_weight"].values
        xs.append(float(bc))
        y_meds.append(_weighted_quantile(vals, wts, 0.50))
        y_los.append(_weighted_quantile(vals, wts, 0.25))
        y_his.append(_weighted_quantile(vals, wts, 0.75))

    x = np.array(xs)
    y_med = np.array(y_meds)
    y_lo = np.array(y_los)
    y_hi = np.array(y_his)

    if len(x) < 2:
        continue

    ax.plot(
        x, y_med,
        color=color_map[alias],
        linewidth=2.5,
        label=alias.replace(" (Inspect)", ""),
        marker=marker_map[alias],
        markersize=8,
        zorder=3,
    )
    ax.fill_between(
        x, y_lo, y_hi,
        color=color_map[alias],
        alpha=0.15,
        zorder=2,
    )

# ── reference line at cost parity ──────────────────────────────────────
ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="AI cost = human cost")

# ── axes formatting ────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xticks(TIME_X_TICKS)
ax.set_xticklabels(TIME_X_LABELS, rotation=30, ha="right")
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlim(1, 1200)

ax.set_yticks(COST_Y_TICKS)
ax.set_yticklabels(COST_Y_LABELS)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_ylim(0.001, 3)

ax.set_xlabel("Task Duration (Human Time)", fontsize=13)
ax.set_ylabel("AI Cost / Human Cost", fontsize=13)
ax.set_title("METR Time Horizon Analysis: AI Cost Ratio by Task Time Horizon (successful attempts only)\n"
             "Excluding tasks < 1 min 30 s", fontsize=15)

# Legend sorted by release date, with IQR explanation
handles, labels = ax.get_legend_handles_labels()
iqr_handle = mpatches.Patch(color="gray", alpha=0.3, label="Weighted IQR (25th\u201375th pctl)")
handles.append(iqr_handle)
labels.append("Weighted IQR (25th\u201375th pctl)")
ax.legend(handles, labels, title="Model", fontsize=11, title_fontsize=12,
          loc="upper left", framealpha=0.9)

fig.tight_layout()
out_path = "cost_ratio_vs_duration.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved to {out_path}")
