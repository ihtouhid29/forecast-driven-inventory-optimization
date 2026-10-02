"""Stage 3: train the demand models and score them against the baselines.

Trains one Tweedie point model and one model per quantile level, all on data
strictly before the validation window, with early stopping on validation.
The 28-day test window is never seen during training or model selection.

Predictions for the test window are saved for Stage 4, which turns them into
order quantities.

Usage:  python scripts/run_forecast.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines as B
from src import config as C
from src import metrics as M
from src import models as MD
from src import viz
from src.data import build_panel, intermittency_profile, revenue_matrix, to_matrix
from src.features import build_features, feature_columns


def _load_features() -> pd.DataFrame:
    cache = C.DATA_PROCESSED / "features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    feats = build_features(build_panel())
    feats.to_parquet(cache, index=False)
    return feats


def fig_importance(imp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    top = imp.head(15).iloc[::-1]
    ax.barh(np.arange(len(top)), top["share"], color=viz.BLUE, height=0.68)
    ax.set_yticks(np.arange(len(top)), top["feature"])
    for i, v in enumerate(top["share"]):
        ax.annotate(f"{v:.1%}", xy=(v, i), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.3, color=viz.INK_MUTED)
    ax.set_xlim(0, top["share"].max() * 1.20)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Share of total split gain")
    ax.set_title("What the model actually uses")
    ax.grid(axis="y", visible=False)
    viz.caption(fig, "Gain-based importance for the point model. Four rolling means and a "
                     "per-item effect take 90% of the gain; every calendar and price feature "
                     "together takes under 4%. The model is largely a learned blend of moving "
                     "averages at different horizons, which is exactly why its margin over the "
                     "moving-average baselines is small.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig7_feature_importance.png")
    plt.close(fig)


def fig_model_vs_baseline(board: pd.DataFrame) -> None:
    b = board.sort_values("test_wrmsse")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = [viz.ORANGE if "lightgbm" in str(n) else viz.BLUE for n in b.index]
    bars = ax.barh(np.arange(len(b)), b["test_wrmsse"], color=colors, height=0.66)
    ax.set_yticks(np.arange(len(b)), b.index)
    ax.set_ylim(len(b) - 0.45, -1.15)
    ax.axvline(1.0, color=viz.INK_MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("1.0 = no better than predicting yesterday",
                xy=(1.0, -0.95), xytext=(6, 0), textcoords="offset points",
                fontsize=8.2, color=viz.INK_MUTED, va="center")
    for bar, v in zip(bars, b["test_wrmsse"]):
        ax.annotate(f"{v:.3f}", xy=(v, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.6, color=viz.INK_MUTED)
    ax.set_xlim(0, max(1.05, b["test_wrmsse"].max() * 1.18))
    ax.set_xlabel("WRMSSE on the 28-day test window (lower is better)")
    ax.set_title("Model against the baselines")
    ax.grid(axis="y", visible=False)
    viz.caption(fig, "The gradient booster has to justify its complexity against methods that "
                     "need no features at all. The margin over Croston and SBA is the part "
                     "the calendar and price features bought.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig8_model_vs_baseline.png")
    plt.close(fig)


def fig_calibration(cal: pd.DataFrame) -> None:
    """Nominal versus achieved coverage: the honesty check on the quantiles."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot([0.45, 1.0], [0.45, 1.0], color=viz.GRID, linewidth=1.6, zorder=1)
    ax.annotate("perfect calibration", xy=(0.72, 0.72), xytext=(6, -14),
                textcoords="offset points", fontsize=8.4, color=viz.INK_MUTED)

    ax.plot(cal["quantile"], cal["model_coverage"], "o-", color=viz.BLUE,
            markersize=7, label="LightGBM quantiles", zorder=3)
    ax.plot(cal["quantile"], cal["baseline_coverage"], "s--", color=viz.ORANGE,
            markersize=6, label="Empirical quantile baseline", zorder=2)

    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Share of actuals at or below the forecast")
    ax.set_title("Quantile calibration on the test window")
    ax.legend(loc="lower right")
    viz.caption(fig, "A point above the line over-covers (too much stock); below, under-covers "
                     "(stockouts). Calibration, not average loss, is what the inventory policy "
                     "in Stage 4 actually depends on.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig9_calibration.png")
    plt.close(fig)


def main() -> int:
    viz.apply_style()
    print("Loading data ...")
    panel = build_panel()
    sales, items, days, first_idx = to_matrix(panel)
    revenue = revenue_matrix(panel, items, days)

    feats = _load_features()
    cols = feature_columns(feats)
    print(f"  {len(feats):,} rows, {len(cols)} features")

    dtrain, dvalid, train_df, valid_df = MD.make_datasets(
        feats, cols, C.VAL_START_DAY, C.TEST_START_DAY)
    print(f"  train {len(train_df):,} rows (days {train_df['day_num'].min()}"
          f"-{train_df['day_num'].max()}), valid {len(valid_df):,} rows")

    test_df = feats[(feats["day_num"] >= C.TEST_START_DAY)
                    & (feats["day_num"] < C.TEST_START_DAY + C.HORIZON)]
    test_days = np.sort(test_df["day_num"].unique())

    # Scale and weights from everything before the test window.
    hist_cols = days < C.TEST_START_DAY
    scale = M.naive_scale(sales[:, hist_cols], valid_from=first_idx)
    weights = M.series_weights(
        np.nan_to_num(revenue[:, hist_cols][:, -28:]).sum(axis=1))
    actual = sales[:, np.isin(days, test_days)]

    # ---- point model -------------------------------------------------------
    print("\nTraining point model (Tweedie) ...")
    t0 = time.time()
    point = MD.train_point(dtrain, dvalid)
    print(f"  best iteration {point.best_iteration}, {time.time() - t0:.0f}s")

    pred_flat = MD.predict(point, test_df, cols)
    pred = MD.to_matrix(test_df, pred_flat, items, test_days)
    model_wrmsse = M.wrmsse(actual, pred, scale, weights)
    model_rmsse = M.rmsse(actual, pred, scale)

    # ---- baselines on the same window --------------------------------------
    history = sales[:, hist_cols]
    board = {}
    for name, fn in B.POINT_BASELINES.items():
        f = fn(history, C.HORIZON)
        board[name] = M.wrmsse(actual, f, scale, weights)
    board["lightgbm_tweedie"] = model_wrmsse
    board = pd.DataFrame({"test_wrmsse": board}).sort_values("test_wrmsse")

    print("\n" + "=" * 62)
    print(f"{'MODEL AGAINST BASELINES (test WRMSSE)':^62}")
    print("=" * 62)
    for name, r in board.iterrows():
        mark = "  <-- model" if name == "lightgbm_tweedie" else ""
        print(f"  {name:<22}{r['test_wrmsse']:>10.4f}{mark}")

    best_base = board.drop(index="lightgbm_tweedie").iloc[0]
    lift = 1.0 - model_wrmsse / best_base["test_wrmsse"]
    print(f"\n  Best baseline: {board.drop(index='lightgbm_tweedie').index[0]} "
          f"({best_base['test_wrmsse']:.4f})")
    print(f"  Model improvement over best baseline: {lift:+.2%}")

    # ---- quantile models ---------------------------------------------------
    print("\nTraining quantile models ...")
    qpreds: dict[float, np.ndarray] = {}
    rows = []
    for q in C.QUANTILES:
        t0 = time.time()
        m = MD.train_quantile(dtrain, dvalid, q)
        p = MD.to_matrix(test_df, MD.predict(m, test_df, cols), items, test_days)
        qpreds[q] = p

        base = B.empirical_quantile(history, C.HORIZON, q)
        rows.append({
            "quantile": q,
            "model_pinball": M.pinball_loss(actual, p, q),
            "baseline_pinball": M.pinball_loss(actual, base, q),
            "model_coverage": M.coverage(actual, p),
            "baseline_coverage": M.coverage(actual, base),
        })
        print(f"  q={q:.3f}  iter {m.best_iteration:>3}  {time.time() - t0:.0f}s"
              f"  pinball {rows[-1]['model_pinball']:.4f}"
              f"  coverage {rows[-1]['model_coverage']:.3f}")

    cal = pd.DataFrame(rows)
    cal["pinball_lift"] = 1.0 - cal["model_pinball"] / cal["baseline_pinball"]

    print("\n" + "=" * 74)
    print(f"{'QUANTILE FORECASTS vs EMPIRICAL BASELINE':^74}")
    print("=" * 74)
    print(f"  {'q':>6}{'model pin':>12}{'base pin':>11}{'lift':>9}"
          f"{'model cov':>12}{'base cov':>11}")
    for _, r in cal.iterrows():
        print(f"  {r['quantile']:>6.3f}{r['model_pinball']:>12.4f}"
              f"{r['baseline_pinball']:>11.4f}{r['pinball_lift']:>+9.1%}"
              f"{r['model_coverage']:>12.3f}{r['baseline_coverage']:>11.3f}")
    print(f"\n  Mean pinball: model {cal['model_pinball'].mean():.4f}  "
          f"baseline {cal['baseline_pinball'].mean():.4f}  "
          f"({1 - cal['model_pinball'].mean() / cal['baseline_pinball'].mean():+.1%})")

    # ---- by demand class ---------------------------------------------------
    prof = intermittency_profile(panel).reindex(items)
    by_class = pd.DataFrame({
        "lightgbm": pd.Series(model_rmsse, index=items),
        "sba": pd.Series(M.rmsse(actual, B.sba(history, C.HORIZON), scale), index=items),
        "seasonal_ma_8w": pd.Series(
            M.rmsse(actual, B.seasonal_moving_average(history, C.HORIZON), scale),
            index=items),
    }).groupby(prof["class"].values, observed=True).median()

    print("\n  Median RMSSE by demand class:")
    print("  " + by_class.round(3).to_string().replace("\n", "\n  "))

    # ---- figures and artefacts ---------------------------------------------
    imp = MD.importance(point, cols, top=20)
    print("\n  Top features by gain:")
    for _, r in imp.head(8).iterrows():
        print(f"    {r['feature']:<22}{r['share']:>7.1%}")

    fig_importance(imp)
    fig_model_vs_baseline(board)
    fig_calibration(cal)

    np.savez_compressed(
        C.DATA_PROCESSED / "forecasts.npz",
        actual=actual, point=pred, items=items.astype(str), days=test_days,
        **{f"q_{q}": v for q, v in qpreds.items()},
    )
    board.to_csv(C.DATA_PROCESSED / "model_scoreboard.csv")
    cal.to_csv(C.DATA_PROCESSED / "quantile_calibration.csv", index=False)
    imp.to_csv(C.DATA_PROCESSED / "feature_importance.csv", index=False)
    by_class.to_csv(C.DATA_PROCESSED / "model_rmsse_by_class.csv")

    print(f"\nForecasts saved for Stage 4. Figures in {C.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
