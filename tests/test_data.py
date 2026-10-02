"""Tests for the data layer.

These are not decoration. The two bugs that silently ruin a forecasting
project are (1) a train/test split that leaks the future into the past and
(2) pre-launch zeros treated as genuine zero demand. Both are tested here.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import (
    build_panel,
    intermittency_profile,
    split_panel,
    trim_prelaunch,
)


@pytest.fixture(scope="module")
def panel():
    return build_panel()


def test_panel_is_complete(panel):
    """Every SKU-day appears exactly once after its launch."""
    assert not panel.duplicated(["item_id", "day_num"]).any()
    assert panel["sales"].notna().all()
    assert (panel["sales"] >= 0).all()


def test_split_is_time_ordered_and_leak_free(panel):
    """Train must end strictly before validation, which must end before test."""
    train, val, test = split_panel(panel)

    assert train["day_num"].max() < val["day_num"].min()
    assert val["day_num"].max() < test["day_num"].min()

    # No date appears in more than one split.
    assert not set(train["date"]) & set(val["date"])
    assert not set(val["date"]) & set(test["date"])

    # The holdout is exactly the competition horizon.
    assert test["day_num"].nunique() == C.HORIZON
    assert val["day_num"].nunique() == C.HORIZON


def test_test_set_is_the_final_days(panel):
    """The holdout is the most recent window, not a random slice."""
    _, _, test = split_panel(panel)
    assert test["date"].max() == panel["date"].max()


def test_trim_prelaunch_removes_leading_unpriced_rows():
    """A SKU that launches late must not contribute pre-launch zeros."""
    df = pd.DataFrame({
        "item_id": ["A"] * 6,
        "day_num": [1, 2, 3, 4, 5, 6],
        "sales": [0, 0, 0, 2, 1, 3],
        "sell_price": [np.nan, np.nan, np.nan, 5.0, 5.0, 5.0],
    })
    out = trim_prelaunch(df)
    assert out["day_num"].min() == 4
    assert len(out) == 3
    assert out["sales"].tolist() == [2, 1, 3]


def test_prices_present_after_trim(panel):
    """After trimming, every remaining row should carry a price."""
    # Weekly price coverage can still have isolated holes; assert the bulk.
    assert panel["sell_price"].notna().mean() > 0.99


def test_intermittency_math():
    """ADI and CV2 computed on a series with known values."""
    # Sales on 3 of 9 days, non-zero sizes 2, 4, 6.
    df = pd.DataFrame({
        "item_id": ["A"] * 9,
        "sales": [0, 2, 0, 0, 4, 0, 0, 6, 0],
    })
    prof = intermittency_profile(df)

    assert prof.loc["A", "adi"] == pytest.approx(9 / 3)

    nz = np.array([2.0, 4.0, 6.0])
    expected_cv2 = (nz.std(ddof=1) / nz.mean()) ** 2
    assert prof.loc["A", "cv2"] == pytest.approx(expected_cv2)

    # ADI 3.0 >= 1.32 and CV2 0.25 < 0.49 -> intermittent
    assert prof.loc["A", "class"] == "intermittent"


def test_panel_is_mostly_intermittent(panel):
    """Sanity check on the headline EDA claim the README quotes."""
    prof = intermittency_profile(panel)
    share = prof["class"].isin(["intermittent", "lumpy"]).mean()
    assert share > 0.80, f"expected a mostly intermittent panel, got {share:.1%}"
