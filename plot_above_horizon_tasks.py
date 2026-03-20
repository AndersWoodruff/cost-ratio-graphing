"""
Scatter plot of GPT-5's passing runs: cost vs task duration,
colored by whether the task sits above or below the 50% time horizon.
"""
import warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

from _alt_common import load_data

# ── data loading ──────────────────────────────────────────────────────
df, _, _, _, _ = load_data()

out_dir = "."

# ── GPT-5: use original Ord's method values ──────────────────────────
alias = "GPT-5 (Inspect)"
agent_df = df[df["alias"] == alias].copy()

CAP = 67.0
horizon_hrs = 1.62
horizon_min = horizon_hrs * 60

# ── Filter to passing runs only ───────────────────────────────────────
pass_df = agent_df[
    (agent_df["score_binarized"] == 1) & (agent_df["estimated_cost"] <= CAP)
].copy()

above = pass_df["human_minutes"] >= horizon_min
below = ~above

n_below = below.sum()
n_above = above.sum()

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    pass_df.loc[below, "estimated_cost"],
    pass_df.loc[below, "human_minutes"],
    s=30, alpha=0.45, color="#2ca02c", edgecolors="none",
    label=f"Below horizon (n={n_below})", zorder=3,
)
ax.scatter(
    pass_df.loc[above, "estimated_cost"],
    pass_df.loc[above, "human_minutes"],
    s=30, alpha=0.45, color="#d62728", edgecolors="none",
    label=f"Above horizon (n={n_above})", zorder=3,
)

# Horizon line
ax.axhline(horizon_min, color="black", linewidth=2, linestyle="--", zorder=4)
ax.text(0.012, horizon_min * 1.08, f"Horizon = {horizon_hrs:.2f} hrs",
        fontsize=12, fontweight="bold", va="bottom")

# Cost cap line
ax.axvline(CAP, color="gray", linewidth=1, linestyle=":", zorder=2)
ax.text(CAP * 0.85, 1.05, f"${CAP:.0f} cap", fontsize=9, color="gray",
        rotation=90, va="bottom", ha="right")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Estimated Cost ($)", fontsize=12)
ax.set_ylabel("Task Duration (Human Time)", fontsize=12)

yticks = [1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1800]
ylabels = ["1m", "2m", "4m", "8m", "15m", "30m", "1h", "2h", "4h", "8h", "16h", "30h"]
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_ylim(1, 2000)
ax.set_xlim(0.01, 100)

ax.legend(fontsize=11, loc="lower right")
ax.set_title(
    f"GPT-5: Passing Tasks Above vs. Below the {horizon_hrs:.2f}h Horizon (${CAP:.0f} cap)",
    fontsize=13, fontweight="bold",
)
ax.grid(True, alpha=0.3, which="major")

fig.tight_layout()
out_path = os.path.join(out_dir, "above_horizon_tasks.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
