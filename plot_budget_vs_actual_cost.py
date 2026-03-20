"""
Multi-model version of the budget-vs-actual-cost chart.

For each model, at its own time horizon:
  - Red bar: the budget threshold (cost cap) needed to achieve that horizon
  - Green bar: the actual median cost of successful tasks at that duration
"""
import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

from _alt_common import load_data, compute_horizon, weighted_median_ci

# ── data loading ──────────────────────────────────────────────────────
df, _, _, headline, df_unfiltered = load_data()

out_dir = "."

HEADLINE_P50 = dict(zip(headline["agent"], headline["p50"]))

abs_thresholds = np.logspace(np.log10(0.01), np.log10(200), 200)

MODELS = [
    "GPT-5 (Inspect)",
    "Claude 4.1 Opus (Inspect)",
    "o3 (Inspect)",
    "Claude 3.7 Sonnet (Inspect)",
    "Claude 3.5 Sonnet (New) (Inspect)",
    "GPT-4o (Inspect)",
]


# ── Compute per-model data ────────────────────────────────────────────
model_results = []

for alias in MODELS:
    # Unfiltered data for horizon fits (matches headline pipeline)
    agent_df_full = df_unfiltered[df_unfiltered["alias"] == alias].copy()
    # Filtered data for cost analysis
    agent_df = df[df["alias"] == alias].copy()
    if len(agent_df_full) == 0:
        continue
    base_scores = agent_df_full["score_binarized"].values.astype(float)
    costs = agent_df_full["estimated_cost"].values
    name = alias.replace(" (Inspect)", "").replace("Claude ", "")

    # Compute horizon curve using unfiltered data
    horizons, h_thresholds = [], []
    for t in abs_thresholds:
        scores = base_scores.copy()
        scores[costs > t] = 0.0
        h = compute_horizon(agent_df_full, scores)
        if h is not None:
            horizons.append(h)
            h_thresholds.append(t)

    if not horizons:
        continue

    # Use METR headline p50 horizon (unrestricted case)
    if alias not in HEADLINE_P50:
        print(f"  Skipping {alias}: no headline p50 found")
        continue
    sat_h = HEADLINE_P50[alias]

    # Find the minimum budget that achieves this horizon
    sat_t = None
    for t_val, h_val in zip(h_thresholds, horizons):
        if h_val >= sat_h * 0.95:  # within 5% of unrestricted
            sat_t = t_val
            break
    if sat_t is None:
        sat_t = h_thresholds[-1]

    # Actual median cost of successful tasks around the horizon duration
    # Use ±0.1 OOM window (like frontier cost plot)
    half_band = 0.1
    log10_h = np.log10(sat_h)
    lo_win = 10 ** (log10_h - half_band)
    hi_win = 10 ** (log10_h + half_band)
    window_mask = (
        (agent_df["human_minutes"] >= lo_win) &
        (agent_df["human_minutes"] <= hi_win) &
        (agent_df["score_binarized"] == 1)
    )
    sub = agent_df[window_mask]
    # Fallback to ±0.2 OOM if too few
    if len(sub) < 3:
        lo_win = 10 ** (log10_h - 0.2)
        hi_win = 10 ** (log10_h + 0.2)
        window_mask = (
            (agent_df["human_minutes"] >= lo_win) &
            (agent_df["human_minutes"] <= hi_win) &
            (agent_df["score_binarized"] == 1)
        )
        sub = agent_df[window_mask]

    if len(sub) < 2:
        print(f"  ERROR: {alias} has fewer than 2 frontier successes "
              f"in [{lo_win:.1f}, {hi_win:.1f}] min window — skipping")
        continue
    else:
        w_costs = sub["estimated_cost"].values
        w_weights = sub["invsqrt_task_weight"].values
        actual_median, ci_lo_med, ci_hi_med = weighted_median_ci(w_costs, w_weights)
        w_normed = w_weights / w_weights.sum()
        actual_mean = np.average(w_costs, weights=w_normed)

        # Per-task hourly rates: cost / (human_minutes / 60)
        hourly_rates = w_costs / (sub["human_minutes"].values / 60.0)
        actual_median_hourly, ci_lo_med_hr, ci_hi_med_hr = weighted_median_ci(hourly_rates, w_weights)
        actual_mean_hourly = np.average(hourly_rates, weights=w_normed)

        # Bootstrap 90% CI on mean (absolute cost and hourly)
        rng = np.random.RandomState(42)
        n = len(w_costs)
        boot_means = []
        boot_means_hr = []
        for _ in range(5000):
            idx = rng.choice(n, size=n, replace=True)
            bw = w_normed[idx]
            bw = bw / bw.sum()
            boot_means.append(np.average(w_costs[idx], weights=bw))
            boot_means_hr.append(np.average(hourly_rates[idx], weights=bw))
        boot_means = np.array(boot_means)
        boot_means_hr = np.array(boot_means_hr)
        ci_lo_mean = np.percentile(boot_means, 5)
        ci_hi_mean = np.percentile(boot_means, 95)
        ci_lo_mean_hr = np.percentile(boot_means_hr, 5)
        ci_hi_mean_hr = np.percentile(boot_means_hr, 95)

    # Format horizon label
    if sat_h >= 60:
        h_label = f"{sat_h / 60:.1f}h"
    else:
        h_label = f"{sat_h:.0f}m"

    model_results.append({
        "name": name,
        "horizon_min": sat_h,
        "horizon_label": h_label,
        "budget": sat_t,
        "actual_median": actual_median,
        "actual_mean": actual_mean,
        "ci_lo_med": ci_lo_med,
        "ci_hi_med": ci_hi_med,
        "ci_lo_mean": ci_lo_mean,
        "ci_hi_mean": ci_hi_mean,
        "actual_median_hourly": actual_median_hourly,
        "actual_mean_hourly": actual_mean_hourly,
        "ci_lo_med_hr": ci_lo_med_hr,
        "ci_hi_med_hr": ci_hi_med_hr,
        "ci_lo_mean_hr": ci_lo_mean_hr,
        "ci_hi_mean_hr": ci_hi_mean_hr,
    })

    print(f"{name}: horizon={h_label}, budget=${sat_t:.2f}, "
          f"actual_median=${actual_median:.2f}, "
          f"ratio={sat_t/actual_median:.1f}x" if actual_median > 0 else "")

# Sort by horizon (strongest to weakest)
model_results.sort(key=lambda r: r["horizon_min"], reverse=True)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x_labels = [f"{r['name']}\n({r['horizon_label']} horizon)" for r in model_results]
# Implied rate: budget / horizon (Ord's method)
budgets_per_hr = [r["budget"] / (r["horizon_min"] / 60.0) for r in model_results]
# Per-task hourly rates (median and mean computed per-task, then aggregated)
means_per_hr = [r["actual_mean_hourly"] for r in model_results]
medians_per_hr = [r["actual_median_hourly"] for r in model_results]

# Bootstrap CIs (already in $/hr)
mean_yerr_lo = [r["actual_mean_hourly"] - r["ci_lo_mean_hr"] for r in model_results]
mean_yerr_hi = [r["ci_hi_mean_hr"] - r["actual_mean_hourly"] for r in model_results]
med_yerr_lo = [r["actual_median_hourly"] - r["ci_lo_med_hr"] for r in model_results]
med_yerr_hi = [r["ci_hi_med_hr"] - r["actual_median_hourly"] for r in model_results]

x_pos = np.arange(len(model_results))
width = 0.25

ax.bar(x_pos - width, budgets_per_hr, width, color="#d62728",
       edgecolor="black", linewidth=0.8,
       label="Implied rate\n(Ord's: budget / horizon)", alpha=0.85)
ax.bar(x_pos, means_per_hr, width, color="#ff7f0e",
       edgecolor="black", linewidth=0.8,
       label="Mean hourly rate\n(per-task)", alpha=0.85,
       yerr=[mean_yerr_lo, mean_yerr_hi], capsize=4, error_kw=dict(ecolor="black", lw=1.2))
ax.bar(x_pos + width, medians_per_hr, width, color="#2ca02c",
       edgecolor="black", linewidth=0.8,
       label="Median hourly rate\n(per-task)", alpha=0.85,
       yerr=[med_yerr_lo, med_yerr_hi], capsize=4, error_kw=dict(ecolor="black", lw=1.2))

for i, (b, mn, md) in enumerate(zip(budgets_per_hr, means_per_hr, medians_per_hr)):
    ax.text(i - width, b * 1.15, f"${b:.1f}/hr", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#d62728")
    ax.text(i, mn * 1.15, f"${mn:.1f}/hr", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#ff7f0e")
    ax.text(i + width, md * 1.15, f"${md:.1f}/hr", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#2ca02c")
    if md > 0:
        factor = b / md
        ax.text(i, max(b, mn, md) * 2.0, f"{factor:.0f}x", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd",
                          edgecolor="#888"))

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("Implied Hourly Rate ($/hr)", fontsize=12)
ax.set_yscale("log")
all_vals = budgets_per_hr + means_per_hr + medians_per_hr
ax.set_ylim(min(all_vals) * 0.3, max(all_vals) * 5)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("Implied Hourly Rate: Ord's Method vs. Actual Task Costs",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
out_path = os.path.join(out_dir, "budget_vs_actual_hourly_rate.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")

# ── Absolute cost version ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))

budgets = [r["budget"] for r in model_results]
means = [r["actual_mean"] for r in model_results]
medians = [r["actual_median"] for r in model_results]

# Bootstrap CIs (absolute)
abs_mean_yerr_lo = [r["actual_mean"] - r["ci_lo_mean"] for r in model_results]
abs_mean_yerr_hi = [r["ci_hi_mean"] - r["actual_mean"] for r in model_results]
abs_med_yerr_lo = [r["actual_median"] - r["ci_lo_med"] for r in model_results]
abs_med_yerr_hi = [r["ci_hi_med"] - r["actual_median"] for r in model_results]

ax2.bar(x_pos - width, budgets, width, color="#d62728",
        edgecolor="black", linewidth=0.8,
        label="Budget threshold needed\n(Ord's method)", alpha=0.85)
ax2.bar(x_pos, means, width, color="#ff7f0e",
        edgecolor="black", linewidth=0.8,
        label="Mean cost of tasks\nat this duration", alpha=0.85,
        yerr=[abs_mean_yerr_lo, abs_mean_yerr_hi], capsize=4, error_kw=dict(ecolor="black", lw=1.2))
ax2.bar(x_pos + width, medians, width, color="#2ca02c",
        edgecolor="black", linewidth=0.8,
        label="Median cost of tasks\nat this duration", alpha=0.85,
        yerr=[abs_med_yerr_lo, abs_med_yerr_hi], capsize=4, error_kw=dict(ecolor="black", lw=1.2))

for i, (b, mn, md) in enumerate(zip(budgets, means, medians)):
    ax2.text(i - width, b * 1.15, f"${b:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color="#d62728")
    ax2.text(i, mn * 1.15, f"${mn:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color="#ff7f0e")
    ax2.text(i + width, md * 1.15, f"${md:.2f}", ha="center", va="bottom",
             fontsize=8, fontweight="bold", color="#2ca02c")
    if md > 0:
        factor = b / md
        ax2.text(i, max(b, mn, md) * 2.0, f"{factor:.0f}x", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd",
                           edgecolor="#888"))

ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, fontsize=10)
ax2.set_ylabel("Cost ($)", fontsize=12)
ax2.set_yscale("log")
all_vals2 = budgets + means + medians
ax2.set_ylim(min(all_vals2) * 0.3, max(all_vals2) * 5)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_title("Budget Required vs. Actual Task Cost at Each Model's Horizon",
              fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

fig2.tight_layout()
out_path2 = os.path.join(out_dir, "budget_vs_actual_cost.png")
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out_path2}")
