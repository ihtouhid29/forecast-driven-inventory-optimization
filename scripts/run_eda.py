"""Exploratory analysis of the M5 panel for one store.

Produces the four figures in reports/figures/ and prints the summary numbers
that the README quotes. Everything here is descriptive: no model is fitted,
no test-set data is touched.

Usage:  python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import viz
from src.data import build_panel, intermittency_profile


def fig_intermittency(panel: pd.DataFrame, prof: pd.DataFrame) -> None:
    """Why this is not a normal forecasting problem."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    # --- Panel A: how much of the panel is zeros -------------------------
    ax = axes[0]
    zero_share = 1.0 - prof["nonzero_share"]
    ax.hist(zero_share, bins=40, color=viz.BLUE, edgecolor=viz.SURFACE, linewidth=0.5)
    med = zero_share.median()
    ax.axvline(med, color=viz.ORANGE, linewidth=2, linestyle="--")
    ax.annotate(f"median {med:.0%}", xy=(med, ax.get_ylim()[1] * 0.88),
                xytext=(8, 0), textcoords="offset points",
                color=viz.ORANGE, fontsize=9, fontweight="bold")
    ax.set_title("A. Most SKUs sell nothing on most days")
    ax.set_xlabel("Share of days with zero sales")
    ax.set_ylabel("Number of SKUs")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    # --- Panel B: Syntetos-Boylan-Croston quadrants ----------------------
    ax = axes[1]
    sub = prof.dropna(subset=["adi", "cv2"])
    ax.scatter(sub["adi"], sub["cv2"], s=5, alpha=0.22,
               color=viz.BLUE, linewidths=0)
    ax.axvline(1.32, color=viz.INK_MUTED, linewidth=1.1)
    ax.axhline(0.49, color=viz.INK_MUTED, linewidth=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ADI  (average days between sales, log scale)")
    ax.set_ylabel("CV²  of non-zero demand size (log scale)")
    ax.set_title("B. Demand class by the SBC scheme")

    counts = sub["class"].value_counts(normalize=True)
    placements = {
        "smooth": (0.06, 0.06), "erratic": (0.06, 0.93),
        "intermittent": (0.72, 0.06), "lumpy": (0.72, 0.93),
    }
    for cls, (fx, fy) in placements.items():
        share = counts.get(cls, 0.0)
        ax.annotate(f"{cls}\n{share:.0%}", xy=(fx, fy), xycoords="axes fraction",
                    fontsize=8.8, color=viz.INK, fontweight="bold",
                    va="top" if fy > 0.5 else "bottom")

    fig.suptitle("Demand is intermittent, which drives every later design choice",
                 x=0.0, ha="left", fontsize=12.5, fontweight="bold")
    viz.caption(fig, "M5 / Walmart, store "
                f"{C.STORE_ID}, {prof.shape[0]:,} SKUs over {panel['day_num'].nunique():,} days. "
                "Lumpy and intermittent series break ordinary point forecasts and mean-based "
                "safety stock, which is why this project forecasts quantiles rather than means.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig1_intermittency.png")
    plt.close(fig)


def fig_seasonality(panel: pd.DataFrame) -> None:
    daily = panel.groupby("date", observed=True)["sales"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    ax.plot(daily.index, daily.values, color=viz.GRID, linewidth=0.8)
    ax.plot(daily.index, daily.rolling(28, center=True).mean(),
            color=viz.BLUE, linewidth=2.0)
    ax.annotate("28-day moving average", xy=(0.62, 0.11), xycoords="axes fraction",
                color=viz.BLUE, fontsize=9, fontweight="bold")
    ax.set_title("A. Store-level demand, daily")
    ax.set_ylabel("Units sold per day")

    ax = axes[1]
    wd = (panel.groupby("wday", observed=True)["sales"].mean()
          .reindex(range(1, 8)))
    labels = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
    bars = ax.bar(labels, wd.values, color=viz.BLUE, width=0.68)
    peak = int(np.nanargmax(wd.values))
    bars[peak].set_color(viz.ORANGE)
    for i, (b, v) in enumerate(zip(bars, wd.values)):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8,
                    color=viz.ORANGE if i == peak else viz.INK_MUTED,
                    fontweight="bold" if i == peak else "normal")
    ax.set_title("B. Weekday profile")
    ax.set_ylabel("Mean units per SKU-day")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Strong weekly seasonality and a slow upward trend",
                 x=0.0, ha="left", fontsize=12.5, fontweight="bold")
    viz.caption(fig, "Weekend demand runs well above midweek. Any forecaster without "
                     "day-of-week features gives that signal away for free.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig2_seasonality.png")
    plt.close(fig)


def fig_snap(panel: pd.DataFrame) -> pd.DataFrame:
    """SNAP benefit days versus ordinary days, by category."""
    g = (panel.groupby(["cat_id", "snap"], observed=True)["sales"]
         .mean().unstack("snap"))
    g.columns = ["non_snap", "snap"]
    g["lift"] = g["snap"] / g["non_snap"] - 1.0
    g = g.sort_values("lift", ascending=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(g))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, g["non_snap"], w, label="Ordinary day", color=viz.BLUE)
    b2 = ax.bar(x + w / 2 + 0.01, g["snap"], w, label="SNAP benefit day", color=viz.ORANGE)

    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=viz.INK_MUTED)

    for i, lift in enumerate(g["lift"]):
        ax.annotate(f"+{lift:.0%}", xy=(i, max(g['snap'].iloc[i], g['non_snap'].iloc[i])),
                    xytext=(0, 17), textcoords="offset points", ha="center",
                    fontsize=9.5, fontweight="bold", color=viz.ORANGE)

    # Headroom so the lift annotations never collide with the title.
    ax.set_ylim(0, max(g["snap"].max(), g["non_snap"].max()) * 1.26)
    ax.set_xticks(x, [c.replace("_", " ").title() for c in g.index])
    ax.set_ylabel("Mean units per SKU-day")
    ax.set_title("SNAP benefit days lift food demand sharply")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    viz.caption(fig, "SNAP disbursement dates are known in advance, so this is a free, "
                     "fully-observable demand driver. It is a calendar feature, not a surprise.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig3_snap_effect.png")
    plt.close(fig)
    return g


def fig_price(panel: pd.DataFrame) -> pd.DataFrame:
    """Demand response to price, measured against each SKU's own baseline.

    A first attempt compared week-on-week changes in price and units. That is
    badly confounded: M5 prices are sticky, so 99% of SKU-weeks land in a
    "no change" bucket, the few change weeks are dominated by trend and
    seasonality, and the estimate came out with the wrong sign for price rises.

    The estimator used here instead indexes each SKU against itself:
        relative price  = price this week / that SKU's median price
        demand index    = units this week / that SKU's median weekly units
    Within-SKU indexing removes the between-SKU differences in price level and
    popularity, which is what the naive week-on-week comparison was picking up.
    This is also exactly the price feature the forecaster uses later, so the
    chart and the model are measuring the same thing.
    """
    wk = (panel.groupby(["item_id", "wm_yr_wk"], observed=True)
          .agg(units=("sales", "sum"), price=("sell_price", "first"))
          .reset_index())
    wk = wk[wk["price"].notna() & (wk["price"] > 0)]

    med = wk.groupby("item_id", observed=True).agg(
        med_price=("price", "median"), med_units=("units", "median"))
    wk = wk.merge(med, on="item_id", how="left")

    # SKUs that essentially never sell carry no information about elasticity.
    wk = wk[wk["med_units"] > 0]

    wk["rel_price"] = wk["price"] / wk["med_price"]
    wk["demand_index"] = wk["units"] / wk["med_units"]

    bins = [0.0, 0.80, 0.90, 0.98, 1.02, 1.10, 1.25, np.inf]
    labels = ["<80%", "80-90%", "90-98%", "98-102%",
              "102-110%", "110-125%", ">125%"]
    wk["bucket"] = pd.cut(wk["rel_price"], bins=bins, labels=labels, right=False)

    agg = wk.groupby("bucket", observed=True)["demand_index"].median().reindex(labels)
    n = wk.groupby("bucket", observed=True).size().reindex(labels)

    # Log-log elasticity on the same indexed quantities, for a single number.
    fit = wk[(wk["demand_index"] > 0) & (wk["rel_price"] > 0)]
    elasticity = np.polyfit(np.log(fit["rel_price"]), np.log(fit["demand_index"]), 1)[0]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = [viz.INK_MUTED if lbl == "98-102%" else viz.ORANGE for lbl in labels]
    bars = ax.bar(labels, agg.values, color=colors, width=0.66)
    ax.axhline(1.0, color=viz.INK_MUTED, linewidth=1.1, zorder=1)

    ax.set_ylim(0, float(np.nanmax(agg.values)) * 1.30)
    for b, v, cnt in zip(bars, agg.values, n.values):
        ax.annotate(f"{v:.2f}\nn={cnt:,}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=viz.INK_MUTED)

    ax.set_ylabel("Median demand index  (1.0 = SKU's typical week)")
    ax.set_xlabel("Price that week, relative to the SKU's own median price")
    ax.set_title("Demand falls as price rises above a SKU's usual level")
    ax.annotate(f"log-log elasticity  {elasticity:.2f}",
                xy=(0.98, 0.94), xycoords="axes fraction", ha="right",
                fontsize=9.5, fontweight="bold", color=viz.ORANGE)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=16, ha="right")
    viz.caption(fig, "Indexing each SKU against its own median removes the popularity and "
                     "price-level differences that confounded a naive week-on-week comparison. "
                     "Relative price is therefore a real feature for the forecaster, and because "
                     "promotions shift the whole demand distribution, it matters for safety stock "
                     "and not only for the point forecast.")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig4_price_effect.png")
    plt.close(fig)
    return agg


def main() -> int:
    viz.apply_style()
    print(f"Building panel for store {C.STORE_ID} ...")
    panel = build_panel()

    print("Profiling intermittency ...")
    prof = intermittency_profile(panel)

    print("\n" + "=" * 62)
    print(f"{'PANEL SUMMARY':^62}")
    print("=" * 62)
    print(f"  SKUs                     {panel['item_id'].nunique():>10,}")
    print(f"  Days                     {panel['day_num'].nunique():>10,}")
    print(f"  SKU-day observations     {len(panel):>10,}")
    print(f"  Share of zero-sale days  {(panel['sales'] == 0).mean():>10.1%}")
    print(f"  Median SKU zero-share    {(1 - prof['nonzero_share']).median():>10.1%}")
    print(f"  Total units              {panel['sales'].sum():>10,}")
    print(f"  Total revenue        ${panel['revenue'].sum():>14,.0f}")

    print("\n  Demand class mix (Syntetos-Boylan-Croston):")
    for cls, share in prof["class"].value_counts(normalize=True).items():
        print(f"    {cls:<14} {share:>6.1%}")

    print("\nDrawing figures ...")
    fig_intermittency(panel, prof)
    fig_seasonality(panel)
    snap = fig_snap(panel)
    price = fig_price(panel)

    print("\n  SNAP lift by category:")
    for cat, row in snap.iterrows():
        print(f"    {str(cat):<12} {row['non_snap']:.3f} -> {row['snap']:.3f}  ({row['lift']:+.1%})")

    print("\n  Median demand index by relative price:")
    for bucket, v in price.items():
        print(f"    {str(bucket):<12} {v:>6.3f}")

    prof.to_parquet(C.DATA_PROCESSED / f"intermittency_{C.STORE_ID}.parquet")
    print(f"\nFigures written to {C.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
