"""Load the M5 competition data and reshape it into a tidy long panel.

Raw M5 ships one row per SKU-store with 1,941 day columns (wide). Every piece
of modelling downstream wants one row per SKU-store-day (long), joined to the
calendar and to the weekly price table. This module does that once and caches
the result as parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_calendar() -> pd.DataFrame:
    """Calendar with weekday, month, year, holiday events and SNAP flags.

    SNAP is the US Supplemental Nutrition Assistance Program. Benefits are
    disbursed on fixed days of the month and food sales spike on those days,
    so the SNAP flag is one of the strongest single features in M5.
    """
    cal = pd.read_csv(C.DATA_RAW / "calendar.csv", parse_dates=["date"])
    cal = cal.sort_values("date").reset_index(drop=True)

    # Some public mirrors of M5 ship the calendar without the `d` column.
    # It is recoverable exactly: the calendar is a contiguous daily series
    # starting at d_1 = FIRST_DATE, so row order gives the day number.
    if "d" not in cal.columns:
        expected_start = pd.Timestamp(C.FIRST_DATE)
        if cal["date"].iloc[0] != expected_start:
            raise ValueError(
                f"calendar.csv starts at {cal['date'].iloc[0].date()}, expected "
                f"{expected_start.date()}; cannot reconstruct day numbers safely."
            )
        gaps = cal["date"].diff().dropna().ne(pd.Timedelta(days=1))
        if gaps.any():
            raise ValueError("calendar.csv is not a contiguous daily series.")
        cal["d"] = "d_" + (np.arange(len(cal)) + 1).astype(str)

    cal["day_num"] = cal["d"].str.removeprefix("d_").astype(int)
    return cal


def load_sales(store_id: str = C.STORE_ID) -> pd.DataFrame:
    """Wide sales table filtered to one store."""
    sales = pd.read_csv(C.DATA_RAW / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"] == store_id].reset_index(drop=True)
    if sales.empty:
        raise ValueError(f"No rows for store_id={store_id!r}. Check the id.")
    return sales


def load_prices(store_id: str = C.STORE_ID) -> pd.DataFrame:
    """Weekly sell prices, filtered to one store.

    Price is NaN before an item is first stocked, which is how M5 encodes
    'this SKU did not exist yet'. We use that to trim leading zeros, which
    matters a lot: treating pre-launch zeros as genuine zero demand badly
    biases both the forecast and the inventory policy.
    """
    prices = pd.read_csv(C.DATA_RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == store_id].reset_index(drop=True)
    return prices


# --------------------------------------------------------------------------
# Reshape
# --------------------------------------------------------------------------
def to_long(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide day columns into one row per SKU-day."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols = [c for c in id_cols if c in sales_wide.columns]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    long = sales_wide.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long["day_num"] = long["d"].str.removeprefix("d_").astype(int)
    long["sales"] = long["sales"].astype(np.int32)

    cal_cols = [
        "d", "date", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    cal_cols = [c for c in cal_cols if c in calendar.columns]
    long = long.merge(calendar[cal_cols], on="d", how="left")
    return long


def attach_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join weekly prices onto the daily panel."""
    return long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def trim_prelaunch(long: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before a SKU's first recorded price.

    Without this, a SKU launched on day 800 looks like it had 799 days of
    zero demand. Any model trained on that learns 'this item never sells',
    and any safety-stock calculation inherits an understated mean and an
    overstated intermittency.
    """
    long = long.sort_values(["item_id", "day_num"])
    has_price = long["sell_price"].notna()
    first_valid = (
        long.loc[has_price]
        .groupby("item_id", observed=True)["day_num"]
        .min()
        .rename("first_day")
    )
    long = long.merge(first_valid, on="item_id", how="left")
    long = long[long["day_num"] >= long["first_day"]].drop(columns=["first_day"])
    return long.reset_index(drop=True)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_panel(store_id: str = C.STORE_ID, cache: bool = True) -> pd.DataFrame:
    """Build (or load from cache) the tidy SKU-day panel for one store."""
    cache_path = C.DATA_PROCESSED / f"panel_{store_id}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    calendar = load_calendar()
    sales = load_sales(store_id)
    prices = load_prices(store_id)

    panel = to_long(sales, calendar)
    panel = attach_prices(panel, prices)
    panel = trim_prelaunch(panel)

    # State-specific SNAP flag, so downstream code does not have to branch.
    state = panel["state_id"].iloc[0]
    panel["snap"] = panel[f"snap_{state}"].astype(np.int8)
    panel = panel.drop(columns=[c for c in panel.columns if c.startswith("snap_")
                                and c != "snap"])

    panel["has_event"] = panel["event_name_1"].notna().astype(np.int8)
    panel["revenue"] = panel["sales"] * panel["sell_price"]

    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        if col in panel.columns:
            panel[col] = panel[col].astype("category")

    if cache:
        panel.to_parquet(cache_path, index=False)
    return panel


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def split_panel(panel: pd.DataFrame):
    """Time-ordered train / validation / test split.

    Split by day number, never randomly. A random split on time series leaks
    the future into the training set and produces scores that collapse the
    moment the model meets real out-of-sample data.
    """
    train = panel[panel["day_num"] < C.VAL_START_DAY]
    val = panel[(panel["day_num"] >= C.VAL_START_DAY)
                & (panel["day_num"] < C.TEST_START_DAY)]
    test = panel[panel["day_num"] >= C.TEST_START_DAY]
    return train, val, test


# --------------------------------------------------------------------------
# Descriptive helpers used by the EDA
# --------------------------------------------------------------------------
def intermittency_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU demand summary: how sparse, how variable, how big.

    ADI  = average inter-demand interval (days between non-zero sales)
    CV2  = squared coefficient of variation of non-zero demand sizes

    The Syntetos-Boylan-Croston scheme classifies series with these two:
        ADI < 1.32 and CV2 < 0.49  -> smooth
        ADI < 1.32 and CV2 >= 0.49 -> erratic
        ADI >= 1.32 and CV2 < 0.49 -> intermittent
        ADI >= 1.32 and CV2 >= 0.49 -> lumpy
    Which class a SKU falls in decides both the right forecaster and the right
    inventory policy, so this is not decoration.
    """
    g = panel.groupby("item_id", observed=True)["sales"]
    n_periods = g.size()
    n_nonzero = g.apply(lambda s: int((s > 0).sum()))

    adi = (n_periods / n_nonzero.replace(0, np.nan)).rename("adi")

    def _cv2(s: pd.Series) -> float:
        nz = s[s > 0]
        if len(nz) < 2 or nz.mean() == 0:
            return np.nan
        return float((nz.std(ddof=1) / nz.mean()) ** 2)

    cv2 = g.apply(_cv2).rename("cv2")

    out = pd.concat(
        [
            adi,
            cv2,
            g.mean().rename("mean_daily"),
            g.sum().rename("total_units"),
            (n_nonzero / n_periods).rename("nonzero_share"),
        ],
        axis=1,
    )

    smooth = (out["adi"] < 1.32) & (out["cv2"] < 0.49)
    erratic = (out["adi"] < 1.32) & (out["cv2"] >= 0.49)
    intermittent = (out["adi"] >= 1.32) & (out["cv2"] < 0.49)
    out["class"] = np.select(
        [smooth, erratic, intermittent],
        ["smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    return out


# --------------------------------------------------------------------------
# Matrix view
# --------------------------------------------------------------------------
def to_matrix(panel: pd.DataFrame):
    """Pivot the panel into a dense (n_series, n_days) sales matrix.

    Baselines and metrics work on whole series at once, so a matrix is the
    natural shape for them; the long panel is the natural shape for feature
    engineering in Stage 3. Both views come from the same source of truth.

    Because SKUs launch at different times, rows are padded with zeros before
    a SKU's launch. `first_day_idx` records where each series really starts so
    that the RMSSE scaling denominator can ignore the padding rather than
    treating it as a long run of genuine zero demand.

    Returns
    -------
    sales : (n_series, n_days) float array
    items : (n_series,) array of item ids, matching row order
    days  : (n_days,) array of day numbers, matching column order
    first_day_idx : (n_series,) column index of each series' first live day
    """
    wide = panel.pivot_table(
        index="item_id", columns="day_num", values="sales",
        aggfunc="sum", observed=True,
    )
    wide = wide.sort_index(axis=0).sort_index(axis=1)

    items = wide.index.to_numpy()
    days = wide.columns.to_numpy()
    live = wide.notna().to_numpy()

    # copy=True is required, not cosmetic. From pandas 3.0 onward to_numpy()
    # may hand back a read-only view of the frame's own buffer under
    # copy-on-write, and writing the zeros below then raises
    # "assignment destination is read-only". Asking for a copy works on
    # every pandas version.
    sales = np.array(wide.to_numpy(dtype=float), copy=True)
    sales[~live] = 0.0

    # First column where each series is live.
    first_day_idx = np.argmax(live, axis=1)
    first_day_idx[~live.any(axis=1)] = live.shape[1]

    return sales, items, days, first_day_idx


def revenue_matrix(panel: pd.DataFrame, items: np.ndarray, days: np.ndarray) -> np.ndarray:
    """(n_series, n_days) revenue, aligned to `to_matrix` row and column order."""
    wide = panel.pivot_table(
        index="item_id", columns="day_num", values="revenue",
        aggfunc="sum", observed=True,
    )
    wide = wide.reindex(index=items, columns=days)
    return np.array(wide.to_numpy(dtype=float), copy=True)
