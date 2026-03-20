"""Shared utilities for alternative time-horizon approaches."""

import json
import sys
import os

import matplotlib
matplotlib.use("Agg")

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "src")
from horizon.utils.logistic import get_x_for_quantile, logistic_regression  # noqa: F401

# ── constants ──────────────────────────────────────────────────────────

OUT_DIR = "reports/time-horizon-1-1/alternative-approaches-v2"
os.makedirs(OUT_DIR, exist_ok=True)

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p", "<", ">", "d", "H"]
Y_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 60, 120, 240, 540, 1440]
Y_LABELS = ["15s", "30s", "1m", "2m", "4m", "8m", "16m", "32m",
            "1h", "2h", "4h", "9h", "1d"]

COST_THRESHOLDS = [None, 1/4, 1/8, 1/16]
QUANTILE = 0.5
REG = 0.1
TRENDLINE_AFTER_DATE = pd.Timestamp("2024-01-01")

REPR_MODELS = ["o3 (Inspect)", "Claude Opus 4.5 (Inspect)"]
FIT_THRESHOLDS = {"1/16x": 1/16, "1/4x": 1/4}

EXCLUDE_AGENTS = set()  # populated by callers if needed

# x-axis ticks for logistic fit plots (log2 scale)
X_TICKS_LOG2 = np.log2([1/60, 0.1, 1, 5, 15, 60, 240, 720, 1800])
X_TICK_LABELS = ["1s", "6s", "1m", "5m", "15m", "1h", "4h", "12h", "30h"]

# Duration buckets for non-parametric analyses
DURATION_BUCKETS = [
    ("<1 min",    0,    1),
    ("1-4 min",   1,    4),
    ("4-16 min",  4,   16),
    ("16-60 min", 16,  60),
    ("1-4 hr",    60, 240),
    ("4-16 hr",  240, 960),
    ("16+ hr",   960, 1e6),
]

# Time axis ticks (log-scaled)
TIME_X_TICKS = [2, 4, 8, 16, 32, 60, 120, 240, 480, 960]
TIME_X_LABELS = ["2 min", "4 min", "8 min", "16 min", "32 min",
                 "1 hr", "2 hr", "4 hr", "8 hr", "16 hr"]

# Cost-ratio y-axis ticks
COST_Y_TICKS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
COST_Y_LABELS = ["0.1%", "0.3%", "1%", "3%", "10%", "30%", "100%", "300%"]

# ── pricing (avg of prompt + completion per token) ────────────────────

PRICE_PER_TOKEN = {
    "Claude 3 Opus (Inspect)": (15 + 75) / 2e6,
    "Claude 3.5 Sonnet (Old) (Inspect)": (6 + 30) / 2e6,
    "Claude 3.5 Sonnet (New) (Inspect)": (6 + 30) / 2e6,
    "Claude 3.7 Sonnet (Inspect)": (3 + 15) / 2e6,
    "Claude 4 Opus (Inspect)": (15 + 75) / 2e6,
    "Claude 4.1 Opus (Inspect)": (15 + 75) / 2e6,
    "Claude Opus 4.5 (Inspect)": (5 + 25) / 2e6,
    "GPT-4 0314": (30 + 60) / 2e6,
    "GPT-4 1106 (Inspect)": (10 + 30) / 2e6,
    "GPT-4 Turbo (Inspect)": (10 + 30) / 2e6,
    "GPT-4o (Inspect)": (2.5 + 10) / 2e6,
    "GPT-5 (Inspect)": (1.25 + 10) / 2e6,
    "GPT-5.1-Codex-Max (Inspect)": (1.25 + 10) / 2e6,
    "GPT-5.2": (1.75 + 14) / 2e6,
    "Gemini 3 Pro": (2 + 12) / 2e6,
    "o1 (Inspect)": (15 + 60) / 2e6,
    "o1-preview": (15 + 60) / 2e6,
    "o3 (Inspect)": (2 + 8) / 2e6,
}

# ── colors ─────────────────────────────────────────────────────────────

_CANONICAL_COLORS = {
    "o1": "#1f77b4", "o1-preview": "#17becf", "o3": "#ff7f0e",
    "o4-mini": "#2ca02c",
    "Claude 3 Opus": "#d62728",
    "Claude 3.5 Sonnet (Old)": "#9467bd",
    "Claude 3.5 Sonnet (New)": "#8c564b",
    "Claude 4 Opus": "#e377c2",
    "Claude 4.1 Opus": "#e377c2",
    "Claude 3.7 Sonnet": "#a65628",
    "Claude Opus 4.5": "#984ea3",
    "GPT-4 1106": "#ff69b4",
    "GPT-4 Turbo": "#4682b4",
    "GPT-4o": "#32cd32",
    "GPT-5": "#7f7f7f",
    "GPT-5.1-Codex-Max": "#5AB2FA",
    "GPT-5.2": "#4AA0E8",
    "Gemini 3 Pro": "#4285F4",
}
_FALLBACK = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)


def get_color(alias):
    canon = alias.replace(" (Inspect)", "").strip()
    return _CANONICAL_COLORS.get(canon, _FALLBACK[hash(canon) % len(_FALLBACK)])


# ── data loading ───────────────────────────────────────────────────────

def load_data():
    rows = []
    with open(r"reports/time-horizon-1-1/data/raw/runs.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df["alias"] != "human"]
    df = df[df["tokens_count"].notna() & (df["tokens_count"] > 0)]

    # Unfiltered: all tasks (matches headline pipeline for horizon fits)
    df_unfiltered = df.copy()

    # Filtered: exclude very short tasks (for cost analysis)
    df = df[df["human_minutes"] >= 1.5]

    # Add cost columns to both
    for d in (df, df_unfiltered):
        d["price_per_token"] = d["alias"].map(PRICE_PER_TOKEN)
    missing_price = df.loc[df["price_per_token"].isna(), "alias"].unique()
    if len(missing_price) > 0:
        raise ValueError(
            f"Models in runs.jsonl have no entry in PRICE_PER_TOKEN and would "
            f"be silently dropped: {sorted(missing_price)}"
        )
    df = df[df["price_per_token"].notna()]
    df_unfiltered = df_unfiltered[df_unfiltered["price_per_token"].notna()]
    for d in (df, df_unfiltered):
        d["estimated_cost"] = d["tokens_count"] * d["price_per_token"]
        d["cost_ratio"] = d["estimated_cost"] / d["human_cost"]

    with open("data/external/release_dates.yaml") as f:
        release_dates = yaml.safe_load(f)["date"]

    aliases = sorted(
        df["alias"].unique(),
        key=lambda a: str(release_dates.get(a, "9999-12-31")),
    )

    headline = pd.read_csv(
        "reports/time-horizon-1-1/data/wrangled/logistic_fits/headline.csv"
    )
    return df, release_dates, aliases, headline, df_unfiltered


# ── helpers ────────────────────────────────────────────────────────────

def fmt_thresh(t):
    if t is None:
        return "unlimited"
    d = int(round(1 / t))
    return f"1/{d}x"


def filter_sota(rdf, after_date=TRENDLINE_AFTER_DATE, exclude_agents=EXCLUDE_AGENTS):
    """SOTA filtering matching METR's get_sota_agents(): keep only models
    whose time_horizon_minutes >= the highest seen so far, chronologically."""
    rdf_after = rdf[rdf["release_date"] >= after_date].sort_values("release_date")
    if exclude_agents:
        rdf_after = rdf_after[~rdf_after["alias"].isin(exclude_agents)]
    sota = []
    highest = float("-inf")
    for row in rdf_after.itertuples():
        if row.time_horizon_minutes >= highest:
            sota.append(row.alias)
            highest = row.time_horizon_minutes
    return rdf_after[rdf_after["alias"].isin(sota)]


def find_rightmost_crossing(x_arr, y_arr, threshold=0.5):
    """Rightmost x where y crosses from >= threshold to < threshold."""
    diff = y_arr - threshold
    for i in range(len(diff) - 2, -1, -1):
        if diff[i] >= 0 and diff[i + 1] < 0:
            x1, x2 = x_arr[i], x_arr[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            return x1 - y1 * (x2 - x1) / (y2 - y1)
    # No downward crossing found
    if np.any(diff >= 0):
        return x_arr[np.where(diff >= 0)[0][-1]]
    return None


def compute_horizon(sub_df, scores, quantile=0.5):
    """Fit logistic on sub_df and return horizon in minutes, or None."""
    if len(sub_df) < 10:
        return None
    weights = sub_df["invsqrt_task_weight"].values.copy()
    weights /= weights.sum()
    log2_min = np.log2(sub_df["human_minutes"].values).reshape(-1, 1)
    try:
        model = logistic_regression(
            log2_min, scores, weights,
            regularization=REG, ensure_weights_sum_to_1=True,
        )
        if model.coef_[0][0] >= 0:
            return None
        h = get_x_for_quantile(model, quantile)
        h_min = 2 ** h
        if h_min < 0.1 or h_min > 1e5:
            return None
        return h_min
    except Exception:
        return None


def weighted_median_ci(values, weights, n_boot=5000, seed=42):
    """Weighted median with 90% bootstrap CI. Returns (median, ci_lo, ci_hi)."""
    weights = weights / weights.sum()
    sort_idx = np.argsort(values)
    sv, sw = values[sort_idx], weights[sort_idx]
    cum_w = np.cumsum(sw)
    median = sv[np.searchsorted(cum_w, 0.5)]
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_medians = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bv, bw = values[idx], weights[idx]
        bw = bw / bw.sum()
        si = np.argsort(bv)
        cum = np.cumsum(bw[si])
        boot_medians.append(bv[si][np.searchsorted(cum, 0.5)])
    boot_medians = np.array(boot_medians)
    ci_lo = np.percentile(boot_medians, 5)
    ci_hi = np.percentile(boot_medians, 95)
    return median, ci_lo, ci_hi


def build_model_release(models, release_dates):
    """Build model_release dict and sorted model list."""
    model_release = {}
    for m in models:
        rd = release_dates.get(m)
        if rd:
            model_release[m] = pd.Timestamp(str(rd))
    sorted_models = sorted(
        [m for m in models if m in model_release],
        key=lambda m: model_release[m],
    )
    return model_release, sorted_models


def plot_horizon_grid(all_results, suptitle, out_path, release_dates):
    """
    2x2 grid of horizon-vs-release-date subplots.

    all_results: dict {threshold_label -> list of {alias, release_date,
                       time_horizon_minutes}}
    """
    # Collect all aliases for consistent markers
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
                ax.set_title("50% Time Horizon with Unlimited Cost")
            else:
                ax.set_title(f"50% Affordable Time Horizon: {thresh_label}")
            continue

        rdf = pd.DataFrame(results).sort_values("release_date")

        for _, row in rdf.iterrows():
            ax.scatter(
                row["release_date"], row["time_horizon_minutes"],
                color=get_color(row["alias"]),
                marker=alias_marker.get(row["alias"], "o"),
                s=100, zorder=5, edgecolors="white", linewidths=0.5,
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
                ha="left", va="top", fontsize=10, color="blue",
            )

        ax.set_yscale("log")
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(Y_LABELS)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(0.1, 1500)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_xlabel("Model Release Date")
        ax.set_ylabel("50% Time Horizon")
        if "unlimited" in thresh_label.lower():
            ax.set_title("50% Time Horizon with Unlimited Cost", fontsize=13)
        else:
            ax.set_title(f"50% Affordable Time Horizon: {thresh_label}", fontsize=13)
        ax.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def setup_fit_axes(ax, log2_minutes):
    """Common formatting for logistic-fit subplots."""
    ax.set_xticks(X_TICKS_LOG2)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=9)
    ax.set_xlim(log2_minutes.min() - 1, log2_minutes.max() + 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Task Duration (Human Time)", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.grid(True, alpha=0.3)


def scatter_data(ax, log2_minutes, scores):
    """Plot pass/fail scatter with jitter."""
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-0.03, 0.03, size=len(scores))
    pass_m = scores == 1.0
    fail_m = scores == 0.0
    frac_m = ~pass_m & ~fail_m
    ax.scatter(log2_minutes[fail_m], scores[fail_m] + jitter[fail_m],
               alpha=0.15, s=12, color="#d62728", zorder=2)
    ax.scatter(log2_minutes[pass_m], scores[pass_m] + jitter[pass_m],
               alpha=0.15, s=12, color="#2ca02c", zorder=2)
    if frac_m.any():
        ax.scatter(log2_minutes[frac_m], scores[frac_m] + jitter[frac_m],
                   alpha=0.3, s=12, color="#ff7f0e", zorder=2)
