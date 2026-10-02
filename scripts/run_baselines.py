"""Stage 2: score every baseline and build the scoreboard.

This establishes the numbers every later model has to beat. Each baseline is
fitted on data strictly before its evaluation window, twice: once fitted on
train and scored on validation, once fitted on train plus validation and
scored on the 28-day test window.

Usage:  python scripts/run_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines as B
from src import config as C
from src import metrics as M
from src import viz
from src.data import build_panel, intermittency_profile, revenue_matrix, to_matrix


# --------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------
def evaluate_window(sales: np.ndarray, revenue: np.ndarray, days: np.ndarray,
                    first_idx: np.ndarray, start_day: int, horizon: int):
    """Score every point baseline on the window beginning at `start_day`.

    Returns the per-series RMSSE for each baseline, plus the weights and
    the actuals, so callers can slice the results any way they like.
    """
    train_cols = days < start_day
    eval_cols = (days >= start_day) & (days < start_day + horizon)

    history = sales[:, train_cols]
    actual = sales[:, eval_cols]
    if actual.shape[1] != horizon:
        raise ValueError(f"expected {horizon} eval days, found {actual.shape[1]}")

    # Scale and weights come from the training window only. Computing either
    # on the evaluation window would leak the answer into the metric.
    scale = M.naive_scale(history, valid_from=first_idx)
    rev_last_28 = np.nan_to_num(revenue[:, train_cols][:, -28:]).sum(axis=1)
    weights = M.series_weights(rev_last_28)

    rows = {}
    for name, fn in B.POINT_BASELINES.items():
        fcst = fn(history, horizon)
        rows[name] = {
            "per_series": M.rmsse(actual, fcst, scale),
            "wrmsse": M.wrmsse(actual, fcst, scale, weights),
            "bias": M.bias(actual, fcst),
            "forecast": fcst,
        }
    return rows, actual, scale, weights, history


def evaluate_quantiles(history: np.ndarray, actual: np.ndarray):
    """Pinball loss and calibration for the empirical-quantile baseline."""
    out = []
    for q in C.QUANTILES:
        fcst = B.empirical_quantile(history, actual.shape[1], q)
        out.append({
            "quantile": q,
            "pinball": M.pinball_loss(actual, fcst, q),
            "coverage": M.coverage(actual, fcst),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_scoreboard(board: pd.DataFrame) -> None:
    """WRMSSE by baseline, test window, sorted best first."""
    b = board.sort_values("test_wrmsse")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    colors = [viz.ORANGE if i == 0 else viz.BLUE for i in range(len(b))]
    bars = ax.barh(np.arange(len(b)), b["test_wrmsse"], color=colors, height=0.66)
    ax.set_yticks(np.arange(len(b)), b.index)
    ax.invert_yaxis()

    ax.axvline(1.0, color=viz.INK_MUTED, linewidth=1.2, linestyle="--")
    # Annotate above the top bar, not beside the bottom one, where it would
    # sit on top of the worst baseline's bar and its value label.
    ax.set_ylim(len(b) - 0.45, -1.15)
    ax.annotate("1.0 = no better than predicting yesterday",
                xy=(1.0, -0.95), xytext=(6, 0), textcoords="offset points",
                fontsize=8.2, color=viz.INK_MUTED, va="center")

    for bar, v in zip(bars, b["test_wrmsse"]):
        ax.annotate(f"{v:.3f}", xy=(v, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.6, color=viz.INK_MUTED)

    ax.set_xlim(0, max(1.05, b["test_wrmsse"].max() * 1.18))
    ax.set_xlabel("WRMSSE on the 28-day test window (lower is better)")
    ax.set_title("Baseline scoreboard")
    ax.grid(axis="y", visible=False)
    viz.caption(fig, "Weighted by each SKU's share of dollar sales in the last 28 training days. "
                     "Any model from Stage 3 onward has to beat the best bar here to justify its "
                     "complexity.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig5_baseline_scoreboard.png")
    plt.close(fig)


def fig_by_class(per_class: pd.DataFrame, chosen: list[str]) -> None:
    """Median RMSSE by demand class, for three contrasting baselines."""
    order = ["smooth", "erratic", "intermittent", "lumpy"]
    order = [c for c in order if c in per_class.index]
    sub = per_class.loc[order, chosen]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(order))
    w = 0.26
    for i, (name, color) in enumerate(zip(chosen, viz.SERIES)):
        off = (i - (len(chosen) - 1) / 2) * (w + 0.02)
        bars = ax.bar(x + off, sub[name], w, label=name.replace("_", " "), color=color)
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.8, color=viz.INK_MUTED)

    ax.axhline(1.0, color=viz.INK_MUTED, linewidth=1.1, linestyle="--")
    ax.set_ylim(0, float(np.nanmax(sub.to_numpy())) * 1.22)
    ax.set_xticks(x, [c.title() for c in order])
    ax.set_ylabel("Median RMSSE (lower is better)")
    ax.set_title("No single baseline wins everywhere")
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="x", visible=False)
    viz.caption(fig, "Different demand classes reward different methods, which is the case for a "
                     "model that can condition on SKU characteristics rather than applying one "
                     "rule to every series.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig6_rmsse_by_class.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    viz.apply_style()
    print("Loading panel ...")
    panel = build_panel()
    sales, items, days, first_idx = to_matrix(panel)
    revenue = revenue_matrix(panel, items, days)

    print("Scoring baselines on validation ...")
    val_rows, _, _, _, _ = evaluate_window(
        sales, revenue, days, first_idx, C.VAL_START_DAY, C.HORIZON)

    print("Scoring baselines on test ...")
    test_rows, test_actual, _, _, test_history = evaluate_window(
        sales, revenue, days, first_idx, C.TEST_START_DAY, C.HORIZON)

    board = pd.DataFrame({
        "val_wrmsse": {k: v["wrmsse"] for k, v in val_rows.items()},
        "test_wrmsse": {k: v["wrmsse"] for k, v in test_rows.items()},
        "test_median_rmsse": {k: np.nanmedian(v["per_series"]) for k, v in test_rows.items()},
        "test_bias": {k: v["bias"] for k, v in test_rows.items()},
    })
    board = board.sort_values("test_wrmsse")

    print("\n" + "=" * 74)
    print(f"{'BASELINE SCOREBOARD':^74}")
    print("=" * 74)
    print(f"  {'baseline':<20}{'val WRMSSE':>12}{'test WRMSSE':>13}"
          f"{'median RMSSE':>14}{'bias':>10}")
    print("  " + "-" * 70)
    for name, r in board.iterrows():
        print(f"  {name:<20}{r['val_wrmsse']:>12.4f}{r['test_wrmsse']:>13.4f}"
              f"{r['test_median_rmsse']:>14.4f}{r['test_bias']:>10.4f}")

    best = board.index[0]
    print(f"\n  Best on test: {best}  (WRMSSE {board.loc[best, 'test_wrmsse']:.4f})")

    # ---- breakdown by demand class ----------------------------------------
    prof = intermittency_profile(panel).reindex(items)
    per_class = pd.DataFrame(
        {name: pd.Series(r["per_series"], index=items).groupby(
            prof["class"].values, observed=True).median()
         for name, r in test_rows.items()}
    )

    print("\n  Median RMSSE by demand class (test):")
    print("  " + per_class.round(3).to_string().replace("\n", "\n  "))

    # ---- quantile baseline -------------------------------------------------
    qdf = evaluate_quantiles(test_history, test_actual)
    print("\n  Empirical-quantile baseline (test):")
    print(f"  {'q':>7}{'pinball':>12}{'coverage':>11}{'gap':>9}")
    for _, r in qdf.iterrows():
        print(f"  {r['quantile']:>7.3f}{r['pinball']:>12.4f}"
              f"{r['coverage']:>11.4f}{r['coverage'] - r['quantile']:>+9.4f}")
    print(f"\n  Mean pinball across quantiles: {qdf['pinball'].mean():.4f}")

    # ---- figures and artefacts --------------------------------------------
    print("\nDrawing figures ...")
    fig_scoreboard(board)

    chosen = [c for c in ["seasonal_naive", "moving_average_28", "sba"]
              if c in per_class.columns]
    fig_by_class(per_class, chosen)

    board.to_csv(C.DATA_PROCESSED / "baseline_scoreboard.csv")
    per_class.to_csv(C.DATA_PROCESSED / "baseline_rmsse_by_class.csv")
    qdf.to_csv(C.DATA_PROCESSED / "baseline_quantiles.csv", index=False)
    np.save(C.DATA_PROCESSED / "test_actual.npy", test_actual)

    print(f"Figures written to {C.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
