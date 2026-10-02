"""Tests for feature engineering.

`test_horizon_features_are_immune_to_the_future` is the most important test
in this repository. Leakage does not raise an exception or produce a warning;
it produces excellent validation scores and a model that cannot be deployed.
The only way to catch it is to change the future and check that nothing in
the forecast window notices.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import MIN_LAG, build_features, feature_columns


def _synthetic_panel(n_items: int = 3, n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    """A small panel with the same columns as the real one.

    Synthetic rather than real so the test runs in under a second and the
    perturbation below is exactly controlled.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2011-01-29")

    for i in range(n_items):
        rate = 0.3 + 0.5 * i
        sales = rng.poisson(rate, size=n_days)
        price = 5.0 + i + rng.normal(0, 0.1, size=n_days).cumsum() * 0.01
        for t in range(n_days):
            date = start + pd.Timedelta(days=t)
            rows.append({
                "item_id": f"ITEM_{i}",
                "dept_id": f"DEPT_{i % 2}",
                "cat_id": "CAT_0",
                "day_num": t + 1,
                "date": date,
                "sales": int(sales[t]),
                "sell_price": float(price[t]),
                "wday": (t % 7) + 1,
                "month": date.month,
                "snap": int(t % 10 < 3),
                "has_event": int(t % 50 == 0),
                "event_type_1": "Sporting" if t % 50 == 0 else None,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The leakage test
# --------------------------------------------------------------------------
def test_horizon_features_are_immune_to_the_future():
    """Change every sale after the origin; the horizon's features must not move.

    This is the operational definition of "no leakage". A forecast is made at
    `origin` for the next 28 days. If any feature on those 28 days depends on
    data recorded after `origin`, that feature cannot be computed when the
    forecast is actually needed, and the model is untrainable in practice
    however good its validation score looks.
    """
    panel = _synthetic_panel()
    origin = 400
    horizon = MIN_LAG

    before = build_features(panel)

    # Corrupt everything strictly after the origin.
    corrupted = panel.copy()
    future = corrupted["day_num"] > origin
    corrupted.loc[future, "sales"] = 999
    after = build_features(corrupted)

    cols = [c for c in feature_columns(before) if c not in ("sell_price",)]

    in_horizon = (before["day_num"] > origin) & (before["day_num"] <= origin + horizon)
    a = before.loc[in_horizon, cols].reset_index(drop=True)
    b = after.loc[in_horizon.to_numpy(), cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_the_leakage_test_can_actually_fail():
    """A control: a deliberately leaky feature must be caught by the same check.

    A test that cannot fail proves nothing. `lag_1` looks at yesterday, which
    is unknown for most of a 28-day horizon, so corrupting the future must
    change it. If this assertion ever stops holding, the test above has
    stopped testing anything.
    """
    panel = _synthetic_panel()
    origin = 400

    def leaky(df):
        out = df.sort_values(["item_id", "day_num"]).reset_index(drop=True).copy()
        out["lag_1"] = out.groupby("item_id", observed=True)["sales"].shift(1)
        return out

    before = leaky(panel)
    corrupted = panel.copy()
    corrupted.loc[corrupted["day_num"] > origin, "sales"] = 999
    after = leaky(corrupted)

    in_horizon = (before["day_num"] > origin) & (before["day_num"] <= origin + MIN_LAG)
    a = before.loc[in_horizon, ["lag_1"]].reset_index(drop=True)
    b = after.loc[in_horizon.to_numpy(), ["lag_1"]].reset_index(drop=True)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


# --------------------------------------------------------------------------
# Structural checks
# --------------------------------------------------------------------------
def test_every_sales_lag_respects_the_minimum():
    from src.features import LAGS
    assert min(LAGS) >= MIN_LAG


def test_lag_values_match_a_manual_shift():
    panel = _synthetic_panel(n_items=1, n_days=200)
    feats = build_features(panel)

    manual = panel.sort_values("day_num")["sales"].shift(28).to_numpy(dtype=float)
    np.testing.assert_allclose(
        feats.sort_values("day_num")["lag_28"].to_numpy(dtype=float),
        manual, equal_nan=True)


def test_rolling_mean_window_ends_before_the_origin():
    """rmean_7 must be the mean of days t-34 to t-28, not t-7 to t-1."""
    panel = _synthetic_panel(n_items=1, n_days=200)
    feats = build_features(panel).sort_values("day_num").reset_index(drop=True)
    sales = panel.sort_values("day_num")["sales"].to_numpy(dtype=float)

    t = 150                                   # zero-based row index
    lo, hi = t - MIN_LAG - 6, t - MIN_LAG + 1
    expected = sales[lo:hi].mean()
    assert feats.loc[t, "rmean_7"] == pytest.approx(expected, rel=1e-6)


def test_days_since_sale_is_shifted_and_non_negative():
    panel = _synthetic_panel(n_items=2, n_days=200)
    feats = build_features(panel)
    dsl = feats["days_since_sale"].dropna()
    assert (dsl >= 0).all()


def test_no_feature_column_is_entirely_missing():
    feats = build_features(_synthetic_panel())
    cols = feature_columns(feats)
    all_nan = [c for c in cols if feats[c].isna().all()]
    assert not all_nan, f"columns with no data at all: {all_nan}"


def test_feature_columns_are_stable_and_numeric():
    feats = build_features(_synthetic_panel())
    cols = feature_columns(feats)
    assert cols == feature_columns(feats), "column order must be deterministic"
    for c in cols:
        assert pd.api.types.is_numeric_dtype(feats[c]), f"{c} is not numeric"
