"""Stage 4: turn the forecasts into orders and price the consequences.

Three ordering policies are simulated against the same realised demand over
the same 28-day test window, from the same opening stock:

  1. historical      textbook reorder point, normal safety factor, no model
  2. forecast_mean   order the expected demand, no safety stock
  3. newsvendor      order the critical-ratio quantile of the forecast

Cost assumptions live in `src/config.py` and are swept in the sensitivity
analysis at the end, because a saving that holds at only one assumed cost
ratio is not a result.

Usage:  python scripts/run_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import inventory as INV
from src import viz
from src.data import build_panel, to_matrix

POLICY_ORDER = ["historical", "forecast_mean", "newsvendor"]
POLICY_LABEL = {
    "historical": "Historical reorder point",
    "forecast_mean": "Forecast mean",
    "newsvendor": "Newsvendor quantile",
}


def load_forecasts():
    z = np.load(C.DATA_PROCESSED / "forecasts.npz", allow_pickle=False)
    quantiles = {float(k[2:]): z[k] for k in z.files if k.startswith("q_")}
    return z["actual"], z["point"], quantiles, z["items"], z["days"]


def build_policies(point, quantiles, costs, history, items):
    """One order-up-to callable per policy, plus the chosen quantile levels."""
    cr = costs.critical_ratio
    levels = INV.pick_quantile_level(cr, list(quantiles))

    # Assemble a per-SKU quantile forecast by picking each SKU's own level.
    picked = np.zeros_like(point)
    for lvl in np.unique(levels):
        rows = levels == lvl
        picked[rows] = quantiles[float(lvl)][rows]

    return {
        "historical": lambda t: INV.order_up_to_historical(history, cr),
        "forecast_mean": lambda t: INV.order_up_to_mean(point, t),
        "newsvendor": lambda t: INV.order_up_to_quantile(point, picked, t),
    }, levels, cr


def fig_cost_breakdown(res: pd.DataFrame) -> None:
    """Stacked cost, and the service level each policy bought."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(res))
    hold = res["holding_cost"].to_numpy()
    short = res["shortage_cost"].to_numpy()
    order = res["ordering_cost"].to_numpy()

    ax.bar(x, hold, 0.6, label="Holding", color=viz.BLUE)
    ax.bar(x, short, 0.6, bottom=hold, label="Shortage", color=viz.ORANGE)
    ax.bar(x, order, 0.6, bottom=hold + short, label="Ordering", color=viz.AQUA)

    totals = hold + short + order
    ax.set_ylim(0, totals.max() * 1.20)
    for i, tot in enumerate(totals):
        ax.annotate(f"${tot:,.0f}", xy=(i, tot), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=viz.INK)

    ax.set_xticks(x, [POLICY_LABEL[p].replace(" ", "\n", 1) for p in res.index])
    ax.yaxis.set_major_formatter(lambda y, _: f"${y:,.0f}")
    ax.set_ylabel("Cost over the 28-day test window")
    ax.set_title("A. Where the money goes")
    ax.set_ylim(0, totals.max() * 1.32)
    ax.legend(loc="upper left", ncols=1, fontsize=8.5)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    fr = res["fill_rate"].to_numpy() * 100
    bars = ax.bar(x, fr, 0.55, color=[viz.BLUE, viz.BLUE, viz.ORANGE])
    ax.set_ylim(min(fr.min() - 4, 85), 100.6)
    for b, v in zip(bars, fr):
        ax.annotate(f"{v:.1f}%", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=viz.INK_MUTED)
    ax.set_xticks(x, [POLICY_LABEL[p].replace(" ", "\n", 1) for p in res.index])
    ax.set_ylabel("Fill rate (% of demanded units served)")
    ax.set_title("B. Service level bought")
    ax.grid(axis="x", visible=False)

    fig.suptitle("The forecast only matters if the decision gets cheaper",
                 x=0.0, ha="left", fontsize=12.5, fontweight="bold")
    viz.caption(fig, "Same realised demand, same opening stock, same cost assumptions. "
                     "The newsvendor policy spends more on holding in order to spend much "
                     "less on shortages, which is the trade the critical ratio is chosen to make.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig10_cost_breakdown.png")
    plt.close(fig)


def fig_sensitivity(sweep: pd.DataFrame) -> None:
    """Does the saving survive different assumptions about shortage cost?"""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    ax.plot(sweep["penalty_mult"], sweep["saving_vs_historical"] * 100,
            "o-", color=viz.BLUE, markersize=7, label="vs historical reorder point")
    ax.plot(sweep["penalty_mult"], sweep["saving_vs_mean"] * 100,
            "s--", color=viz.ORANGE, markersize=6, label="vs forecast mean")
    ax.axhline(0, color=viz.INK_MUTED, linewidth=1.1)

    ax.set_xlabel("Stockout penalty (multiple of lost margin)")
    ax.set_ylabel("Cost saving from the newsvendor policy (%)")
    ax.set_title("The result holds across the assumed cost of a stockout")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:+.0f}%")
    ax.legend(loc="best")
    viz.caption(fig, "M5 supplies no cost of goods or stockout cost, so the headline saving "
                     "depends on assumptions. Sweeping the one that matters most shows whether "
                     "the conclusion is real or an artefact of a single chosen number.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig11_sensitivity.png")
    plt.close(fig)


def main() -> int:
    viz.apply_style()
    print("Loading forecasts and prices ...")
    actual, point, quantiles, items, test_days = load_forecasts()

    panel = build_panel()
    sales, all_items, all_days, _ = to_matrix(panel)
    hist_cols = all_days < C.TEST_START_DAY
    history = sales[:, hist_cols]

    prices = (panel.groupby("item_id", observed=True)["sell_price"]
              .mean().reindex(pd.Index(all_items)).to_numpy())
    costs = INV.build_cost_model(prices)

    policies, levels, cr = build_policies(point, quantiles, costs, history, items)

    print(f"  {len(items):,} SKUs, {actual.shape[1]} days")
    print(f"  Critical ratio: median {np.median(cr):.3f}, "
          f"range {cr.min():.3f} to {cr.max():.3f}")
    print(f"  Quantile levels chosen: "
          f"{dict(zip(*np.unique(levels, return_counts=True)))}")
    print(f"  Protection interval: {C.PROTECTION_DAYS} days "
          f"(review {C.REVIEW_PERIOD_DAYS} + lead {C.LEAD_TIME_DAYS})")

    # ---- run the three policies -------------------------------------------
    print("\nSimulating ...")
    rows = {}
    for name in POLICY_ORDER:
        r = INV.simulate(actual, policies[name], costs)
        rows[name] = {
            "total_cost": r.total_cost,
            "holding_cost": r.holding_cost,
            "shortage_cost": r.shortage_cost,
            "ordering_cost": r.ordering_cost,
            "fill_rate": r.fill_rate,
            "cycle_service": r.cycle_service,
            "units_short": r.units_short,
        }
        print(f"  {name:<16} total ${r.total_cost:>10,.0f}   "
              f"fill {r.fill_rate:>6.2%}   short {r.units_short:>8,.0f} units")

    res = pd.DataFrame(rows).T.loc[POLICY_ORDER]

    base = res.loc["historical", "total_cost"]
    nv = res.loc["newsvendor", "total_cost"]
    fm = res.loc["forecast_mean", "total_cost"]

    print("\n" + "=" * 70)
    print(f"{'POLICY COMPARISON (28-day test window)':^70}")
    print("=" * 70)
    print(f"  {'policy':<24}{'total $':>12}{'holding':>11}{'shortage':>11}"
          f"{'fill':>9}")
    for name, r in res.iterrows():
        print(f"  {POLICY_LABEL[name]:<24}{r['total_cost']:>12,.0f}"
              f"{r['holding_cost']:>11,.0f}{r['shortage_cost']:>11,.0f}"
              f"{r['fill_rate']:>9.2%}")

    print(f"\n  Newsvendor vs historical reorder point: "
          f"{1 - nv / base:+.1%} cost, "
          f"{(res.loc['newsvendor', 'fill_rate'] - res.loc['historical', 'fill_rate']) * 100:+.2f} pts fill rate")
    print(f"  Newsvendor vs forecast mean:            "
          f"{1 - nv / fm:+.1%} cost, "
          f"{(res.loc['newsvendor', 'fill_rate'] - res.loc['forecast_mean', 'fill_rate']) * 100:+.2f} pts fill rate")

    # ---- sensitivity to the stockout assumption ---------------------------
    print("\nSweeping the stockout penalty assumption ...")
    sweep_rows = []
    for mult in [0.5, 1.0, 1.5, 2.5, 4.0, 6.0]:
        c = INV.build_cost_model(prices, penalty_mult=mult)
        pol, _, _ = build_policies(point, quantiles, c, history, items)
        costs_by = {n: INV.simulate(actual, pol[n], c).total_cost for n in POLICY_ORDER}
        sweep_rows.append({
            "penalty_mult": mult,
            "critical_ratio": float(np.median(c.critical_ratio)),
            **{f"cost_{n}": v for n, v in costs_by.items()},
            "saving_vs_historical": 1 - costs_by["newsvendor"] / costs_by["historical"],
            "saving_vs_mean": 1 - costs_by["newsvendor"] / costs_by["forecast_mean"],
        })
        print(f"  penalty x{mult:<4}  CR {sweep_rows[-1]['critical_ratio']:.3f}   "
              f"saving vs historical {sweep_rows[-1]['saving_vs_historical']:+.1%}   "
              f"vs mean {sweep_rows[-1]['saving_vs_mean']:+.1%}")

    sweep = pd.DataFrame(sweep_rows)

    fig_cost_breakdown(res)
    fig_sensitivity(sweep)

    res.to_csv(C.DATA_PROCESSED / "policy_comparison.csv")
    sweep.to_csv(C.DATA_PROCESSED / "policy_sensitivity.csv", index=False)
    print(f"\nFigures in {C.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
