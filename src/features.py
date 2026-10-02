"""Feature engineering for the demand model.

The single most important design decision in this file is the **minimum lag
of 28 days**, and it is worth stating plainly because it is what keeps the
whole project honest.

The forecast is made once, at an origin, for the following 28 days. A feature
is only legitimate if its value is knowable at that origin. Consider a feature
built from yesterday's sales, used to predict day origin+20: on the day the
forecast is actually made, "yesterday" relative to origin+20 has not happened
yet. Training on it produces a model that scores beautifully in validation and
cannot be run in production, because the input does not exist when you need it.

So every sales-derived feature here is shifted by at least `MIN_LAG` = 28 days,
the length of the horizon. Then for any day in the horizon, the feature's
source data falls on or before the origin. One model, trained once, forecasts
all 28 days directly. No recursion, no leakage, and `tests/test_features.py`
asserts it mechanically rather than trusting this docstring.

Calendar and price features are exempt, and legitimately so: the M5 calendar
is published years ahead and shelf prices for the forecast weeks are given.
Those are known at the origin by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# The horizon. Every sales-derived feature is shifted by at least this much.
MIN_LAG = C.HORIZON

LAGS = [28, 35, 42, 49, 56, 364]
ROLL_WINDOWS = [7, 28, 56, 180]

CATEGORICAL = ["item_code", "dept_code", "cat_code", "wday", "month", "event_type_code"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _codes(s: pd.Series) -> np.ndarray:
    """Integer codes for a categorical column, stable across calls."""
    return pd.Categorical(s).codes.astype(np.int16)


def _days_since_last_sale(sales: np.ndarray) -> np.ndarray:
    """Per-row count of days since the previous non-zero sale.

    Vectorised over a (n_series, n_days) matrix. A running index of the last
    day with demand, forward-filled, subtracted from the current day.
    """
    n_series, n_days = sales.shape
    cols = np.arange(n_days)[None, :]
    last = np.where(sales > 0, cols, -1)
    last = np.maximum.accumulate(last, axis=1)
    out = np.where(last >= 0, cols - last, n_days).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the panel with model features attached.

    Rows keep their original identity columns so that predictions can be
    joined back to (item_id, day_num) for the inventory stages.
    """
    df = panel.sort_values(["item_id", "day_num"]).reset_index(drop=True).copy()

    # ---- identity and calendar (known at the origin) ----------------------
    df["item_code"] = _codes(df["item_id"])
    df["dept_code"] = _codes(df["dept_id"])
    df["cat_code"] = _codes(df["cat_id"])
    df["event_type_code"] = _codes(df["event_type_1"].fillna("none"))

    df["day_of_month"] = df["date"].dt.day.astype(np.int8)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(np.int8)   # M5: 1=Sat, 2=Sun

    g = df.groupby("item_id", observed=True)

    # ---- price features (known at the origin) -----------------------------
    df["price_median"] = g["sell_price"].transform("median")
    df["price_rel"] = (df["sell_price"] / df["price_median"]).astype(np.float32)
    df["price_chg_1w"] = g["sell_price"].transform(
        lambda s: s / s.shift(7) - 1.0).astype(np.float32)
    df["price_rank"] = g["sell_price"].transform(
        lambda s: s.rank(pct=True)).astype(np.float32)

    # ---- sales lags -------------------------------------------------------
    for lag in LAGS:
        assert lag >= MIN_LAG, f"lag {lag} would leak inside a {MIN_LAG}-day horizon"
        df[f"lag_{lag}"] = g["sales"].shift(lag).astype(np.float32)

    # ---- rolling statistics, all computed on the already-shifted series ---
    # Shift first, then roll. Rolling first and shifting after would let the
    # window straddle the origin.
    shifted = g["sales"].shift(MIN_LAG)
    df["_shifted"] = shifted

    gs = df.groupby("item_id", observed=True)["_shifted"]
    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = gs.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()).astype(np.float32)
    for w in [28, 56]:
        df[f"rstd_{w}"] = gs.transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()).astype(np.float32)
        df[f"rmax_{w}"] = gs.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).max()).astype(np.float32)
        df[f"nonzero_share_{w}"] = gs.transform(
            lambda s, w=w: s.gt(0).rolling(w, min_periods=1).mean()).astype(np.float32)

    # Demand size given a sale: the numerator of Croston's ratio, as a feature.
    df["mean_size_56"] = (df["rmean_56"] /
                          df["nonzero_share_56"].replace(0, np.nan)).astype(np.float32)

    # ---- days since last sale, computed on the shifted series -------------
    wide = df.pivot_table(index="item_id", columns="day_num", values="sales",
                          aggfunc="sum", observed=True).sort_index(axis=1)
    items = wide.index.to_numpy()
    days = wide.columns.to_numpy()
    mat = np.nan_to_num(wide.to_numpy(dtype=float))

    dsl = _days_since_last_sale(mat)
    dsl_df = pd.DataFrame(dsl, index=items, columns=days).stack().rename("_dsl_raw")
    dsl_df.index.names = ["item_id", "day_num"]
    df = df.merge(dsl_df.reset_index(), on=["item_id", "day_num"], how="left")

    # Shift it, like every other sales-derived feature.
    df["days_since_sale"] = df.groupby("item_id", observed=True)["_dsl_raw"] \
                              .shift(MIN_LAG).astype(np.float32)

    df = df.drop(columns=["_shifted", "_dsl_raw", "price_median"])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns handed to the model, in a stable order."""
    cols = [
        "item_code", "dept_code", "cat_code",
        "wday", "month", "day_of_month", "week_of_year", "is_weekend",
        "snap", "has_event", "event_type_code",
        "sell_price", "price_rel", "price_chg_1w", "price_rank",
        "days_since_sale", "mean_size_56",
    ]
    cols += [f"lag_{l}" for l in LAGS]
    cols += [f"rmean_{w}" for w in ROLL_WINDOWS]
    cols += [f"rstd_{w}" for w in [28, 56]]
    cols += [f"rmax_{w}" for w in [28, 56]]
    cols += [f"nonzero_share_{w}" for w in [28, 56]]
    return [c for c in cols if c in df.columns]


def training_frame(df: pd.DataFrame, min_day: int | None = None) -> pd.DataFrame:
    """Rows usable for training: features present, and optionally recent only.

    The longest lag is 364 days, so the first year of every SKU has no usable
    long lag. Rather than drop those rows entirely (which would throw away the
    short-history SKUs the model most needs to learn about), we require only
    that the 28-day lag exists and let LightGBM handle the remaining NaNs,
    which it does natively by learning a default direction for missing values.
    """
    out = df[df["lag_28"].notna()]
    if min_day is not None:
        out = out[out["day_num"] >= min_day]
    return out
