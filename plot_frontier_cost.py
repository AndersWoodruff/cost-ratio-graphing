"""
Frontier cost scatter: does reaching a longer time-horizon require more
inference spend?

X-axis: model's 50 % time horizon (from logistic fit of score ~ log2(minutes))
Y-axis: median AI-cost / human-cost for attempts within ±0.1 OOM of that frontier

Produces two plots:
  1. Successes only (original)
  2. All attempts (including failures)

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
    load_data, build_model_release,
    TIME_X_TICKS, TIME_X_LABELS, COST_Y_TICKS, COST_Y_LABELS,
)


def weighted_percentile(values, weights, q):
    """Return the weighted q-th percentile (q in 0–1)."""
    sort_idx = np.argsort(values)
    sv, sw = values[sort_idx], weights[sort_idx]
    sw = sw / sw.sum()
    cum = np.cumsum(sw)
    return sv[np.searchsorted(cum, q)]


def weighted_median_iqr(values, weights):
    """Weighted median with IQR (25th–75th percentile)."""
    median = weighted_percentile(values, weights, 0.50)
    q25 = weighted_percentile(values, weights, 0.25)
    q75 = weighted_percentile(values, weights, 0.75)
    return median, q25, q75

# ── load data ──────────────────────────────────────────────────────────

_, release_dates, _, headline, df = load_data()
MODELS = df["alias"].unique().tolist()
model_release, sorted_models = build_model_release(MODELS, release_dates)

p50_map = dict(zip(headline["agent"], headline["p50"]))


# ── gather results ────────────────────────────────────────────────────

def gather_results(df, successes_only=True):
    label = "successes only" if successes_only else "all attempts"
    results = []

    for alias in sorted_models:
        adf = df[df["alias"] == alias].copy()
        if len(adf) < 20:
            continue

        p50_minutes = p50_map.get(alias)
        if p50_minutes is None or p50_minutes < 1 or p50_minutes > 5000:
            continue

        half_band = 0.1
        log10_p50 = np.log10(p50_minutes)
        lo = 10 ** (log10_p50 - half_band)
        hi = 10 ** (log10_p50 + half_band)

        frontier = adf[(adf["human_minutes"] >= lo) & (adf["human_minutes"] <= hi)]
        if successes_only:
            frontier = frontier[frontier["score_binarized"] == 1]

        # Widen to ±0.2 OOM if too few
        if len(frontier) < 3:
            lo = 10 ** (log10_p50 - 0.2)
            hi = 10 ** (log10_p50 + 0.2)
            frontier = adf[(adf["human_minutes"] >= lo) & (adf["human_minutes"] <= hi)]
            if successes_only:
                frontier = frontier[frontier["score_binarized"] == 1]

        if len(frontier) < 5:
            print(f"  WARNING ({label}): {alias} has only {len(frontier)} frontier "
                  f"points — skipping")
            continue

        ratios = frontier["cost_ratio"].values
        weights = frontier["invsqrt_task_weight"].values
        median_cost_ratio, q25, q75 = weighted_median_iqr(ratios, weights)

        short = alias.replace(" (Inspect)", "").strip()
        results.append({
            "alias": alias,
            "short": short,
            "p50_minutes": p50_minutes,
            "median_cost_ratio": median_cost_ratio,
            "q25": q25,
            "q75": q75,
            "release_date": model_release[alias],
            "n_frontier": len(frontier),
        })
        print(f"  {short:30s}  p50={p50_minutes:8.1f} min  "
              f"cost_ratio={median_cost_ratio:.4f}  "
              f"IQR=[{q25:.4f}, {q75:.4f}]  "
              f"({len(frontier)} frontier {label} in "
              f"[{lo:.1f}, {hi:.1f}] min)")

    return pd.DataFrame(results)


# ── plot ──────────────────────────────────────────────────────────────

def make_plot(rdf, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(13, 8))

    date_nums = np.array([d.toordinal() for d in rdf["release_date"]])
    norm = mcolors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
    cmap = mcm.plasma

    # Error bars (IQR: 25th–75th percentile)
    for _, row in rdf.iterrows():
        color = cmap(norm(row["release_date"].toordinal()))
        ax.plot(
            [row["p50_minutes"], row["p50_minutes"]],
            [row["q25"], row["q75"]],
            color=color, linewidth=2, alpha=0.5, zorder=4,
            solid_capstyle="round",
        )
        cap_w = 0.04
        for yval in [row["q25"], row["q75"]]:
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

    # # Trend line (log-log linear regression → power law)
    # log_x = np.log10(rdf["p50_minutes"].values)
    # log_y = np.log10(rdf["median_cost_ratio"].values)
    # reg = LinearRegression().fit(log_x.reshape(-1, 1), log_y)
    # slope = reg.coef_[0]
    # r2 = reg.score(log_x.reshape(-1, 1), log_y)
    # x_line = np.linspace(log_x.min() - 0.15, log_x.max() + 0.15, 200)
    # y_line = 10 ** reg.predict(x_line.reshape(-1, 1))
    # ax.plot(
    #     10 ** x_line, y_line,
    #     color="black", linestyle=":", linewidth=2, alpha=0.7, zorder=2,
    # )
    # ax.annotate(
    #     f"slope = {slope:.2f}\nR\u00b2 = {r2:.2f}",
    #     xy=(0.02, 0.02), xycoords="axes fraction",
    #     ha="left", va="bottom", fontsize=11, fontweight="bold",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
    #               edgecolor="black", alpha=0.85),
    # )

    # Colourbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.4, aspect=15, anchor=(0.0, 1.0))
    cbar.set_label("Release Date", fontsize=9)
    unique_dates = sorted(rdf["release_date"].unique())
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
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    iqr_handle = plt.Line2D([0], [0], color="gray", linewidth=2, alpha=0.5)
    handles.append(iqr_handle)
    labels.append("IQR (25th\u201375th percentile)")
    ax.legend(handles, labels, loc="upper right", fontsize=10)

    # Grid
    ax.grid(False, which="both")
    for yval in COST_Y_TICKS:
        ax.axhline(y=yval, color="grey", linewidth=0.5, alpha=0.3, zorder=0)
    for xval in TIME_X_TICKS:
        ax.axvline(x=xval, color="grey", linewidth=0.5, alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved to {out_path}")


# ── Plot 1: successes only (original) ────────────────────────────────

print("=== Successes only ===")
rdf_success = gather_results(df, successes_only=True)
if len(rdf_success):
    make_plot(
        rdf_success,
        title="METR Time Horizon Analysis: Cost at 50% Time Horizon",
        ylabel="AI Cost / Human Cost (at Frontier ±0.1 OOM)",
        out_path="frontier_cost.png",
    )

# ── Plot 2: all attempts (including failures) ────────────────────────

print("\n=== All attempts (including failures) ===")
rdf_all = gather_results(df, successes_only=False)
if len(rdf_all):
    make_plot(
        rdf_all,
        title="METR Time Horizon Analysis: Cost at 50% Time Horizon (Incl. Failures)",
        ylabel="AI Cost / Human Cost (at Frontier ±0.1 OOM, all attempts)",
        out_path="frontier_cost_all.png",
    )
