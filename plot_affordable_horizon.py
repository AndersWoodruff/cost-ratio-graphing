"""
Factored model: 2x2 grids at 50% and 80% time horizons.

Decompose: P(pass AND affordable) = P(pass | dur) * P(affordable | pass, dur)

P(pass | dur) is always fit on the full pipeline dataset (runs.jsonl) so it
matches headline.csv exactly.  For cost-thresholded cases the same P(pass)
curve is multiplied by P(affordable | pass, dur), which is fit on the
_alt_common dataset that carries price/cost columns.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression

from _alt_common import (
    load_data, get_color, find_rightmost_crossing,
    filter_sota, fmt_thresh,
    logistic_regression, get_x_for_quantile,
    OUT_DIR, REG, MARKERS, Y_TICKS, Y_LABELS, TRENDLINE_AFTER_DATE,
)

# ── data loading ──────────────────────────────────────────────────────

# Full pipeline data — used for P(pass | dur) fits (matches headline.csv)
df_full = pd.read_json(
    "reports/time-horizon-1-1/data/raw/runs.jsonl",
    lines=True, orient="records", convert_dates=False,
)

# _alt_common data with price/cost columns — used for P(affordable | pass)
df_cost, release_dates, aliases, headline, _ = load_data()

# Headline logistic fits — canonical p50/p80 for the unlimited-cost case
headline_horizon = {
    0.5: dict(zip(headline["agent"], headline["p50"])),
    0.8: dict(zip(headline["agent"], headline["p80"])),
}

# Aliases from the full dataset (excluding human)
aliases_all = sorted(
    df_full["alias"].unique(),
    key=lambda a: str(release_dates.get(a, "9999-12-31")),
)
aliases_all = [a for a in aliases_all if a != "human"]


# ── P(pass | dur) — fit once per agent on the full pipeline data ──────

pass_models = {}   # alias → (model, log2_min_range)
for alias in aliases_all:
    agent_df = df_full[df_full["alias"] == alias]
    if len(agent_df) < 10:
        continue
    weights = agent_df["invsqrt_task_weight"].values.copy()
    weights /= weights.sum()
    scores = agent_df["score_binarized"].values.astype(float)
    log2_min = np.log2(agent_df["human_minutes"].values).reshape(-1, 1)
    try:
        model = logistic_regression(
            log2_min, scores, weights,
            regularization=REG, ensure_weights_sum_to_1=True,
        )
        if model.coef_[0][0] >= 0:
            continue
        pass_models[alias] = (model, float(log2_min.min()), float(log2_min.max()))
    except Exception:
        continue

print(f"Fit P(pass) for {len(pass_models)} agents")


# ── factored horizon using pre-computed P(pass) ──────────────────────

_FAIL = (None, None, None, None, None, None, None)


def compute_factored_horizon(alias, cost_thresh, quantile):
    """
    Returns (horizon_minutes, model_pass, model_afford, x_range,
             p_pass, p_afford, p_joint) or Nones on failure.

    P(pass) comes from pre-computed pass_models (full pipeline data).
    P(afford | pass) is fit on df_cost (which has cost_ratio).
    """
    if alias not in pass_models:
        return _FAIL
    model_pass, xmin, xmax = pass_models[alias]
    x_r = np.linspace(xmin - 1, xmax + 1, 500)

    if cost_thresh is None:
        # Unlimited — just use P(pass)
        try:
            pq_log2 = get_x_for_quantile(model_pass, quantile)
            pq_min = 2 ** pq_log2
            if pq_min < 0.1 or pq_min > 1e5:
                return None, model_pass, None, None, None, None, None
            p_p = model_pass.predict_proba(x_r.reshape(-1, 1))[:, 1]
            return pq_min, model_pass, None, x_r, p_p, None, None
        except Exception:
            return None, model_pass, None, None, None, None, None

    # Cost-thresholded — multiply P(pass) by P(afford | pass)
    agent_cost_df = df_cost[df_cost["alias"] == alias]
    if len(agent_cost_df) < 10:
        return None, model_pass, None, None, None, None, None

    pass_mask = agent_cost_df["score_binarized"].values == 1
    if pass_mask.sum() < 10:
        return None, model_pass, None, None, None, None, None

    pass_df = agent_cost_df[pass_mask]
    affordable = (pass_df["cost_ratio"].values <= cost_thresh).astype(float)

    p_p = model_pass.predict_proba(x_r.reshape(-1, 1))[:, 1]

    if affordable.sum() == 0:
        return None, model_pass, None, None, None, None, None
    if affordable.sum() == len(affordable):
        # All passes are affordable → P(afford|pass) = 1, joint = P(pass)
        try:
            pq_log2 = get_x_for_quantile(model_pass, quantile)
            pq_min = 2 ** pq_log2
            if pq_min < 0.1 or pq_min > 1e5:
                return None, model_pass, None, None, None, None, None
            return pq_min, model_pass, None, x_r, p_p, np.ones_like(p_p), p_p
        except Exception:
            return None, model_pass, None, None, None, None, None

    log2_min_pass = np.log2(pass_df["human_minutes"].values).reshape(-1, 1)
    weights_pass = pass_df["invsqrt_task_weight"].values.copy()
    weights_pass /= weights_pass.sum()

    try:
        model_afford = logistic_regression(
            log2_min_pass, affordable, weights_pass,
            regularization=REG, ensure_weights_sum_to_1=True,
        )
    except Exception:
        return None, model_pass, None, None, None, None, None

    p_a = model_afford.predict_proba(x_r.reshape(-1, 1))[:, 1]
    p_j = p_p * p_a

    h_log2 = find_rightmost_crossing(x_r, p_j, quantile)
    if h_log2 is None:
        return None, model_pass, model_afford, x_r, p_p, p_a, p_j

    h_min = 2 ** h_log2
    if h_min < 0.1 or h_min > 1e5:
        return None, model_pass, model_afford, x_r, p_p, p_a, p_j

    return h_min, model_pass, model_afford, x_r, p_p, p_a, p_j


# ── parameterized grid plotting ───────────────────────────────────────

def plot_horizon_grid(all_results, suptitle, out_path, release_dates, pct_label):
    """2x2 grid of horizon-vs-release-date subplots."""
    all_aliases = set()
    for results in all_results.values():
        for r in results:
            all_aliases.add(r["alias"])
    sorted_aliases = sorted(
        all_aliases,
        key=lambda a: str(release_dates.get(a, "9999-12-31")),
    )
    alias_marker = {a: MARKERS[i % len(MARKERS)]
                    for i, a in enumerate(sorted_aliases)}

    sns.set_theme(style="whitegrid")
    n_panels = len(all_results)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 7 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (thresh_label, results) in enumerate(all_results.items()):
        ax = axes_flat[idx]
        if not results:
            ax.text(0.5, 0.5, "No valid models", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            if "unlimited" in thresh_label.lower():
                ax.set_title(f"{pct_label} Time Horizon with Unlimited Cost")
            else:
                ax.set_title(f"{pct_label} Affordable Time Horizon: {thresh_label}")
            continue

        rdf = pd.DataFrame(results).sort_values("release_date")

        for _, row in rdf.iterrows():
            ax.scatter(
                row["release_date"], row["time_horizon_minutes"],
                color=get_color(row["alias"]),
                marker=alias_marker.get(row["alias"], "o"),
                s=140, zorder=5, edgecolors="white", linewidths=0.5,
                label=row["alias"].replace(" (Inspect)", ""),
            )

        rdf_fit = filter_sota(rdf)
        if len(rdf_fit) >= 2:
            X_d = np.array(
                [mdates.date2num(d) for d in rdf_fit["release_date"]]
            ).reshape(-1, 1)
            y_l = np.log(rdf_fit["time_horizon_minutes"].clip(1e-3).values)
            reg = LinearRegression().fit(X_d, y_l)
            r2 = reg.score(X_d, y_l)
            coef = reg.coef_[0]
            doubling = np.log(2) / coef if coef > 1e-10 else float("inf")

            x_line = np.linspace(X_d.min() - 60, X_d.max() + 180, 200)
            y_line = np.exp(reg.predict(x_line.reshape(-1, 1)))
            ax.plot(
                [mdates.num2date(d) for d in x_line], y_line,
                color="blue", linewidth=2, alpha=0.6,
            )
            ax.annotate(
                f"Doubling: {doubling / 30.44:.1f} months\nR\u00b2={r2:.2f}",
                xy=(0.02, 0.98), xycoords="axes fraction",
                ha="left", va="top", fontsize=14, color="blue",
            )

        ax.set_yscale("log")
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(Y_LABELS, fontsize=13)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(0.1, 1500)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
        ax.set_xlabel("Model Release Date", fontsize=14)
        ax.set_ylabel(f"{pct_label} Time Horizon", fontsize=14)
        if "unlimited" in thresh_label.lower():
            ax.set_title(f"{pct_label} Time Horizon with Unlimited Cost", fontsize=16)
        else:
            ax.set_title(f"{pct_label} Affordable Time Horizon: {thresh_label}", fontsize=16)
        ax.legend(fontsize=9, loc="lower right")

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── compute horizons and generate grids ───────────────────────────────

COST_THRESHOLDS_EXT = [None, 1/4, 1/8, 1/16, 1/32]

for quantile, pct_label, suffix in [(0.5, "50%", ""), (0.8, "80%", "_p80")]:
    print(f"\n{'='*60}")
    print(f"  Computing {pct_label} time horizons")
    print(f"{'='*60}")

    all_aliases_set = set()
    all_results = {}
    for cost_thresh in COST_THRESHOLDS_EXT:
        label = f"AI cost < {fmt_thresh(cost_thresh)} human cost"
        results = []
        for alias in aliases_all:
            if cost_thresh is None:
                # Use canonical headline figures for unlimited cost
                h_min = headline_horizon[quantile].get(alias)
                if h_min is None or h_min < 0.1 or h_min > 1e5:
                    continue
            else:
                h_min = compute_factored_horizon(alias, cost_thresh, quantile)[0]
            if h_min is None:
                continue
            rd = release_dates.get(alias)
            if rd is None:
                continue
            rd_ts = pd.Timestamp(str(rd))
            if rd_ts < TRENDLINE_AFTER_DATE:
                continue
            results.append({
                "alias": alias,
                "release_date": rd_ts,
                "time_horizon_minutes": h_min,
            })
            all_aliases_set.add(alias)
            print(f"  [{fmt_thresh(cost_thresh)}] {alias:40s} horizon={h_min:.1f} min")
        all_results[label] = results

    # 2x2 grid (unlimited, 1/4x, 1/8x, 1/32x) — skip 1/16x
    _grid_keys = [k for k, t in zip(all_results.keys(), COST_THRESHOLDS_EXT) if t != 1/16]
    grid_results = {k: all_results[k] for k in _grid_keys}
    plot_horizon_grid(
        grid_results,
        f"Factored Model: {pct_label} Affordable Time Horizon",
        f"affordable_horizon_grid{suffix}.png",
        release_dates,
        pct_label,
    )


# ── fit visualization: one page per model ─────────────────────────────

from _alt_common import setup_fit_axes, scatter_data, X_TICKS_LOG2, X_TICK_LABELS

QUANTILE = 0.5  # horizon line for fit plots

FIT_MODELS = [
    "o3 (Inspect)",
    "Claude Opus 4.5 (Inspect)",
    "Claude 4 Opus (Inspect)",
    "GPT-5 (Inspect)",
    "Claude 3.7 Sonnet (Inspect)",
    "o1 (Inspect)",
]
FIT_MODELS = [m for m in FIT_MODELS if m in pass_models]

FIT_THRESH = {
    "1/32x": 1/32,
    "1/16x": 1/16,
    "1/8x": 1/8,
    "1/4x": 1/4,
}

for alias in FIT_MODELS:
    # Scatter data comes from df_cost (has cost_ratio for filtering)
    agent_cost_df = df_cost[df_cost["alias"] == alias].copy()
    if len(agent_cost_df) < 10:
        continue
    log2_min_scatter = np.log2(agent_cost_df["human_minutes"].values)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for col, (t_label, t_val) in enumerate(FIT_THRESH.items()):
        ax = axes_flat[col]

        # Scatter: green = pass AND affordable, red = fail OR too expensive
        scores_filtered = agent_cost_df["score_binarized"].values.copy().astype(float)
        scores_filtered[agent_cost_df["cost_ratio"].values > t_val] = 0.0
        scatter_data(ax, log2_min_scatter, scores_filtered)

        # Curves from the corrected factored model
        h_min, mp, ma, x_r, p_p, p_a, p_j = compute_factored_horizon(
            alias, t_val, QUANTILE,
        )

        if x_r is not None and p_p is not None:
            ax.plot(x_r, p_p, color="#1f77b4", linewidth=2.5, label="P(pass)")
        if x_r is not None and p_a is not None:
            ax.plot(x_r, p_a, color="#2ca02c", linewidth=2.5,
                    linestyle="--", label="P(afford|pass)")
        if x_r is not None and p_j is not None:
            ax.plot(x_r, p_j, color="#d62728", linewidth=2.5,
                    label="P(afford & pass)")

        ax.axhline(y=QUANTILE, color="gray", linestyle=":", linewidth=1, alpha=0.5)

        if h_min is not None:
            h_log2 = np.log2(h_min)
            ax.axvline(x=h_log2, color="black", linestyle="--",
                       linewidth=1.5, alpha=0.7)
            ax.annotate(
                f"Horizon = {h_min:.1f} min",
                xy=(h_log2, QUANTILE), xytext=(h_log2 + 0.5, QUANTILE + 0.15),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray"),
            )

        setup_fit_axes(ax, log2_min_scatter)
        ax.set_title(f"Cost < {t_label} Human Cost", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)

    short_name = alias.replace(" (Inspect)", "")
    fig.suptitle(
        f"Factored Model for {short_name}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe_name = short_name.replace(" ", "_").replace(".", "").lower()
    out = f"affordable_horizon_fit_{safe_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
