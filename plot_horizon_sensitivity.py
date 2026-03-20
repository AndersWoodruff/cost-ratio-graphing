"""
Sensitivity analysis: what actually drives the time horizon higher?

Counterfactual decomposition for GPT-5:
For each cost threshold step, leave out each bucket's newly-unlocked runs,
refit the logistic, and measure how much the p50 actually drops.

Two-panel plot:
  Top: the horizon curve (black line, log y-axis)
  Bottom: stacked percentage breakdown showing what fraction of the
          cumulative horizon gain is attributable to each duration bucket
"""
import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

from _alt_common import load_data, compute_horizon

# ── data loading ──────────────────────────────────────────────────────
df, _, _, _, df_unfiltered = load_data()

out_dir = "."

BUCKETS = [
    ("<1m",    0,      1),
    ("1-4m",   1,      4),
    ("4-16m",  4,     16),
    ("16-60m", 16,    60),
    ("1-4h",   60,   240),
    ("4-16h",  240,  960),
    ("16+h",   960, 1e6),
]
BUCKET_COLORS = ["#17becf", "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

abs_thresholds = np.logspace(np.log10(0.01), np.log10(200), 80)


# ── GPT-5 only ────────────────────────────────────────────────────────
alias = "GPT-5 (Inspect)"
# Unfiltered data for horizon fits (matches headline pipeline)
agent_df = df_unfiltered[df_unfiltered["alias"] == alias].copy()
base_scores = agent_df["score_binarized"].values.astype(float)
costs = agent_df["estimated_cost"].values

bmasks = []
for _, lo, hi in BUCKETS:
    bmasks.append(
        (agent_df["human_minutes"].values >= lo) & (agent_df["human_minutes"].values < hi)
    )

# ── Step 1: full horizon at each threshold ────────────────────────────
full_horizons = []
all_filtered_scores = []

for t in abs_thresholds:
    scores = base_scores.copy()
    scores[costs > t] = 0.0
    all_filtered_scores.append(scores)
    full_horizons.append(compute_horizon(agent_df, scores))

# ── Step 2: counterfactual decomposition ──────────────────────────────
contributions = {b[0]: [] for b in BUCKETS}

for i in range(1, len(abs_thresholds)):
    h_curr = full_horizons[i]
    h_prev = full_horizons[i - 1]

    if h_curr is None or h_prev is None:
        for b in BUCKETS:
            contributions[b[0]].append(0.0)
        continue

    delta_h = h_curr - h_prev
    t_prev, t_curr = abs_thresholds[i - 1], abs_thresholds[i]
    newly_unlocked = (costs > t_prev) & (costs <= t_curr)

    bucket_contribs = []
    for bi, (bname, _, _) in enumerate(BUCKETS):
        bucket_newly_unlocked = newly_unlocked & bmasks[bi]
        if not bucket_newly_unlocked.any():
            bucket_contribs.append(0.0)
            continue
        counterfactual_scores = all_filtered_scores[i].copy()
        counterfactual_scores[bucket_newly_unlocked] = 0.0
        h_without = compute_horizon(agent_df, counterfactual_scores)
        if h_without is None:
            bucket_contribs.append(delta_h)
        else:
            bucket_contribs.append(h_curr - h_without)

    # No rescaling: leave-one-out contributions from a nonlinear model
    # don't sum to delta_h due to interaction effects — that's expected.
    for bi, (bname, _, _) in enumerate(BUCKETS):
        contributions[bname].append(bucket_contribs[bi])

# ── Step 3: cumulative percentage attribution ─────────────────────────
contrib_thresholds = abs_thresholds[1:]

# Use cumulative sums instead of per-step values: naturally smooth
# without an ad-hoc rolling kernel, and robust to small negative blips.
cum_contribs = {}
for bname in [b[0] for b in BUCKETS]:
    cum_contribs[bname] = np.cumsum(contributions[bname])

# Percentage of total cumulative gain at each step
pct = {b[0]: np.zeros(len(contrib_thresholds)) for b in BUCKETS}
for i in range(len(contrib_thresholds)):
    total = sum(max(cum_contribs[b[0]][i], 0) for b in BUCKETS)
    if total > 1e-6:
        for b in BUCKETS:
            pct[b[0]][i] = 100.0 * max(cum_contribs[b[0]][i], 0) / total

# ── Plot: two panels sharing x-axis ──────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
)

# -- Top panel: horizon curve ------------------------------------------
valid_h = [(t, h) for t, h in zip(abs_thresholds, full_horizons) if h is not None]
th, hh = zip(*valid_h)
ax_top.plot(th, hh, "k-", linewidth=2.5, zorder=5)

ax_top.set_yscale("log")
ax_top.set_ylabel("50% Time Horizon", fontsize=12)
yticks = [1/4, 1/2, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960]
ylabels = ["15s", "30s", "1m", "2m", "4m", "8m", "15m", "30m",
           "1h", "2h", "4h", "8h", "16h"]
ax_top.set_yticks(yticks)
ax_top.set_yticklabels(ylabels)
ax_top.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_top.set_ylim(0.3, 1200)
ax_top.set_title("GPT-5: What Drives the Time Horizon Higher?",
                 fontsize=13, fontweight="bold")
ax_top.grid(True, alpha=0.3, which="major")

# -- Bottom panel: stacked percentage attribution ----------------------
stack_data = [np.clip(pct[b[0]], 0, None) for b in BUCKETS]
ax_bot.stackplot(
    contrib_thresholds, *stack_data,
    labels=[b[0] for b in BUCKETS],
    colors=BUCKET_COLORS, alpha=0.7,
)

ax_bot.set_xscale("log")
ax_bot.set_xlabel("Absolute Cost Threshold ($)", fontsize=12)
ax_bot.set_ylabel("% of Cumulative Gain", fontsize=12)
ax_bot.set_ylim(0, 100)
ax_bot.set_xlim(0.1, 200)
ax_bot.legend(fontsize=8, loc="upper left", ncol=3)
ax_bot.grid(True, alpha=0.3)

fig.tight_layout()
out_path = os.path.join(out_dir, "horizon_sensitivity.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
