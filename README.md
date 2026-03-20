# Standalone cost-analysis plots

These scripts extend the METR time-horizon analysis with cost-related visualisations. Run any of them from the repo root (`python plot_*.py`).

## Scripts → outputs

| Script | Output |
|---|---|
| `plot_affordable_horizon.py` | `affordable_horizon_grid.png`, `affordable_horizon_grid_p80.png`, `affordable_horizon_fit_*.png` |
| `plot_above_horizon_tasks.py` | `above_horizon_tasks.png` |
| `plot_cost_ratio_vs_duration.py` | `cost_ratio_vs_duration.png` |
| `plot_budget_vs_actual_cost.py` | `budget_vs_actual_hourly_rate.png`, `budget_vs_actual_cost.png` |
| `plot_frontier_cost.py` | `frontier_cost.png` |
| `plot_frontier_cost_variants.py` | `frontier_cost_p50_band*.png`, `frontier_cost_p80_band*.png` |
| `plot_horizon_sensitivity.py` | `horizon_sensitivity.png` |

## Data setup

These scripts expect to live inside (or alongside) a checkout of the METR `eval-analysis-public` repo. They read:

- `reports/time-horizon-1-1/data/raw/runs.jsonl` — per-run results
- `reports/time-horizon-1-1/data/wrangled/logistic_fits/headline.csv` — pre-computed logistic fits
- `data/external/release_dates.yaml` — model release dates

All shared helpers and token-pricing constants live in `_alt_common.py`.
