"""Stage 5: optimise ordering under a shared capacity constraint.

Stage 4's newsvendor treats every SKU independently, which is right until
something shared runs out. Here a weekly capacity limit binds, the SKUs
compete, and two ways of resolving that competition are compared on identical
cost accounting:

  * proportional rationing  -- cut every request by the same fraction
  * CP-SAT lot-sizing       -- decide jointly which SKU gets the space and when

The capacity is swept from slack to tight, because the interesting question
is not "does optimisation help" but "how tight does it have to get before it
starts to matter".

Usage:  python scripts/run_lotsizing.py
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
from src import lotsizing as LS
from src import viz
from src.data import build_panel, to_matrix

# Scope. CLSP is NP-hard; 300 SKUs over 4 weekly periods is 2,400 integer
# variables, which solves in seconds and is large enough that the structure
# is real rather than a toy.
N_SKUS = 300
N_PERIODS = 4
PERIOD_DAYS = 7
TIME_LIMIT = 60.0

# Capacity as a fraction of EXPECTED demand over the window.
#
# A first version set this against the newsvendor quantile plan instead. That
# plan asks for 2.14x expected demand, so "70% capacity" was really 150% of
# what was needed, and every result was measured in a regime where nothing
# was actually scarce. Anchoring on expected demand makes 100% the point
# where capacity exactly covers the mean, which is where the trade-off
# genuinely starts.
CAPACITY_RATIOS = [1.50, 1.20, 1.00, 0.85, 0.70]


def to_periods(daily: np.ndarray) -> np.ndarray:
    """Aggregate a daily matrix into weekly buckets."""
    n_items, n_days = daily.shape
    usable = N_PERIODS * PERIOD_DAYS
    return daily[:, :usable].reshape(n_items, N_PERIODS, PERIOD_DAYS).sum(axis=2)


def fig_capacity_sweep(sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    ax.plot(sweep["capacity_ratio"], sweep["heuristic_cost"], "s--",
            color=viz.ORANGE, markersize=6.5, label="Proportional rationing")
    ax.plot(sweep["capacity_ratio"], sweep["optimiser_cost"], "o-",
            color=viz.BLUE, markersize=7, label="CP-SAT lot-sizing")
    ax.invert_xaxis()
    ax.set_xlabel("Capacity as a share of expected demand")
    ax.set_ylabel("Total cost over 28 days ($)")
    ax.set_title("A. Cost as capacity tightens")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="upper left")

    ax = axes[1]
    saving = sweep["saving"] * 100
    colors = [viz.ORANGE if v > 0 else viz.GRID for v in saving]
    bars = ax.bar(np.arange(len(sweep)), saving, 0.6, color=colors)
    pad = max(abs(saving.min()), abs(saving.max())) * 0.30
    ax.set_ylim(min(saving.min() - pad, -pad), saving.max() + pad)
    for b, v in zip(bars, saving):
        ax.annotate(f"{v:+.1f}%", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=8.6, color=viz.INK_MUTED)
    ax.set_xticks(np.arange(len(sweep)),
                  [f"{r:.0%}" for r in sweep["capacity_ratio"]])
    ax.set_xlabel("Capacity as a share of expected demand")
    ax.set_ylabel("Cost saving from optimising (%)")
    ax.axhline(0, color=viz.INK_MUTED, linewidth=1.1)
    ax.set_title("B. What optimisation is worth")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Optimisation only pays once capacity is genuinely scarce",
                 x=0.0, ha="left", fontsize=12.5, fontweight="bold")
    viz.caption(fig, f"{N_SKUS} highest-revenue SKUs, {N_PERIODS} weekly periods, identical cost "
                     "accounting. The crossover near 100% of expected demand is the finding: with "
                     "room to spare, carrying the newsvendor buffer beats optimising allocation, "
                     "because deterministic lot-sizing holds no safety stock of its own. Below "
                     "that, the optimiser's ability to choose which SKU to short and to shift a "
                     "purchase into an adjacent week is worth real money.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig12_capacity_sweep.png")
    plt.close(fig)


def main() -> int:
    viz.apply_style()
    print("Loading forecasts ...")
    z = np.load(C.DATA_PROCESSED / "forecasts.npz", allow_pickle=False)
    actual, point = z["actual"], z["point"]
    quantiles = {float(k[2:]): z[k] for k in z.files if k.startswith("q_")}

    panel = build_panel()
    sales, all_items, all_days, _ = to_matrix(panel)
    prices = (panel.groupby("item_id", observed=True)["sell_price"]
              .mean().reindex(pd.Index(all_items)).to_numpy())
    costs = INV.build_cost_model(prices)

    # ---- scope to the SKUs that matter -------------------------------------
    revenue = (sales[:, all_days < C.TEST_START_DAY][:, -180:].sum(axis=1)
               * np.nan_to_num(prices))
    top = np.argsort(revenue)[::-1][:N_SKUS]
    print(f"  {N_SKUS} highest-revenue SKUs "
          f"({revenue[top].sum() / revenue.sum():.1%} of store revenue)")

    # ---- per-SKU newsvendor target, as in Stage 4 --------------------------
    cr = costs.critical_ratio
    levels = INV.pick_quantile_level(cr, list(quantiles))
    picked = np.zeros_like(point)
    for lvl in np.unique(levels):
        rows = levels == lvl
        picked[rows] = quantiles[float(lvl)][rows]

    demand_p = to_periods(actual[top])        # what really happened
    plan_q = to_periods(picked[top])          # what each SKU independently wants
    plan_point = to_periods(point[top])       # the unbiased demand estimate
    holding_p = costs.holding_per_day[top] * PERIOD_DAYS
    shortage_p = costs.shortage[top]

    expected = plan_point.sum(axis=0)
    print(f"  Expected weekly demand:   {expected.round().astype(int)} units")
    print(f"  Newsvendor weekly ask:    {plan_q.sum(axis=0).round().astype(int)} units "
          f"({plan_q.sum() / plan_point.sum():.2f}x expected)")

    # The optimiser plans against the POINT forecast; the heuristic rations
    # the newsvendor quantile plan. They are given the same capacity and
    # scored on the same realised demand.
    #
    # Two earlier formulations failed, both instructively.
    #
    # (1) Capacity anchored to the newsvendor ask rather than to expected
    #     demand. The ask is 2.16x expected, so "70% capacity" was really
    #     150% of what was needed and nothing was ever scarce.
    #
    # (2) Optimiser handed the quantile plan as its demand. Worse, and for a
    #     deeper reason: shortage cost is linear in the model, so charging p
    #     per unit short of an inflated target makes the last unit of safety
    #     stock look exactly as valuable as the first. The problem collapses
    #     into a knapsack, the solver fills expensive SKUs to their inflated
    #     target and zeroes cheap ones entirely, and those zeroed SKUs then
    #     miss their real demand completely. Uniform rationing beat it badly.
    #
    # The fix is to let the optimiser plan against the unbiased forecast,
    # where a unit of shortage really does cost p. The consequence, visible
    # in the crossover below, is that the deterministic CLSP carries no
    # safety stock of its own and therefore loses while capacity is slack.
    # That is a real limitation of deterministic lot-sizing, not a bug.

    rows = []
    for ratio in CAPACITY_RATIOS:
        cap = expected * ratio

        heur = LS.proportional_rationing(plan_q, cap)
        h_eval = LS.evaluate_plan(heur, demand_p, holding_p, shortage_p,
                                  C.ORDER_LINE_COST)

        res = LS.solve_clsp(
            demand=plan_point, holding=holding_p, shortage_cost=shortage_p,
            setup_cost=C.ORDER_LINE_COST, capacity=cap,
            time_limit=TIME_LIMIT, workers=2,
        )
        o_eval = LS.evaluate_plan(res.orders, demand_p, holding_p, shortage_p,
                                  C.ORDER_LINE_COST)

        saving = 1.0 - o_eval["total_cost"] / h_eval["total_cost"]
        rows.append({
            "capacity_ratio": ratio,
            "heuristic_cost": h_eval["total_cost"],
            "optimiser_cost": o_eval["total_cost"],
            "saving": saving,
            "heuristic_fill": h_eval["fill_rate"],
            "optimiser_fill": o_eval["fill_rate"],
            "heuristic_lines": h_eval["order_lines"],
            "optimiser_lines": o_eval["order_lines"],
            "status": res.status,
            "gap": res.gap,
            "solve_seconds": res.wall_time,
        })
        print(f"  capacity {ratio:>5.0%}   heuristic ${h_eval['total_cost']:>9,.0f}   "
              f"optimiser ${o_eval['total_cost']:>9,.0f}   "
              f"saving {saving:>+6.1%}   [{res.status}, gap {res.gap:.2%}, "
              f"{res.wall_time:.0f}s]")

    sweep = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"{'CAPACITATED LOT-SIZING: OPTIMISER vs RATIONING':^78}")
    print("=" * 78)
    print(f"  {'capacity':>9}{'heuristic $':>13}{'optimiser $':>13}{'saving':>9}"
          f"{'heur fill':>11}{'opt fill':>10}{'status':>11}")
    for _, r in sweep.iterrows():
        print(f"  {r['capacity_ratio']:>9.0%}{r['heuristic_cost']:>13,.0f}"
              f"{r['optimiser_cost']:>13,.0f}{r['saving']:>+9.1%}"
              f"{r['heuristic_fill']:>11.2%}{r['optimiser_fill']:>10.2%}"
              f"{r['status']:>11}")

    tight = sweep.iloc[-1]
    dfill = (tight["optimiser_fill"] - tight["heuristic_fill"]) * 100
    print(f"\n  At {tight['capacity_ratio']:.0%} capacity the optimiser costs "
          f"{tight['saving']:.1%} less, at a fill rate {abs(dfill):.2f} points "
          f"{'higher' if dfill >= 0 else 'lower'}.")

    crossover = sweep[sweep["saving"] > 0]
    if not crossover.empty:
        print(f"  Optimisation starts paying below "
              f"{crossover['capacity_ratio'].max():.0%} of expected demand.")

    fig_capacity_sweep(sweep)
    sweep.to_csv(C.DATA_PROCESSED / "lotsizing_sweep.csv", index=False)
    print(f"\nFigures in {C.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
