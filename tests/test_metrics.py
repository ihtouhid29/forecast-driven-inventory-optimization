"""Tests for the metrics.

Hand-computed values, not regression snapshots. If a metric is wrong, every
number in the project is wrong and nothing downstream would notice, so these
are checked against arithmetic done by hand in the comments.
"""
import numpy as np
import pytest

from src import metrics as M


# --------------------------------------------------------------------------
# Scaling denominator
# --------------------------------------------------------------------------
def test_naive_scale_hand_computed():
    """Series [1, 2, 4, 4]: diffs 1, 2, 0 -> squares 1, 4, 0 -> 5/3."""
    hist = np.array([[1.0, 2.0, 4.0, 4.0]])
    assert M.naive_scale(hist)[0] == pytest.approx(5 / 3)


def test_naive_scale_ignores_prelaunch_padding():
    """Zeros before launch must not inflate the observation count.

    Row is [0, 0, 1, 2, 4, 4] with the series starting at index 2. Masked, the
    usable diffs are the same 1, 2, 0 as above, so the scale stays 5/3.

    Unmasked, the padding contributes two extra diffs (0 -> 0 and 0 -> 1):
    squares 0, 1, 1, 4, 0 over five observations, giving 6/5 = 1.2. Smaller
    than 5/3, which is the problem: a deflated denominator flatters every
    RMSSE computed against it, and the flattery lands hardest on exactly the
    late-launching SKUs that are hardest to forecast.
    """
    hist = np.array([[0.0, 0.0, 1.0, 2.0, 4.0, 4.0]])
    scaled = M.naive_scale(hist, valid_from=np.array([2]))
    assert scaled[0] == pytest.approx(5 / 3)

    unmasked = M.naive_scale(hist)
    assert unmasked[0] == pytest.approx(6 / 5)
    assert unmasked[0] < scaled[0]


def test_naive_scale_flat_series_is_nan():
    """A series that never changes has no naive error; it cannot be scaled."""
    hist = np.array([[3.0, 3.0, 3.0, 3.0]])
    assert np.isnan(M.naive_scale(hist)[0])


# --------------------------------------------------------------------------
# RMSSE
# --------------------------------------------------------------------------
def test_rmsse_perfect_forecast_is_zero():
    actual = np.array([[2.0, 5.0, 1.0]])
    scale = np.array([1.5])
    assert M.rmsse(actual, actual.copy(), scale)[0] == pytest.approx(0.0)


def test_rmsse_hand_computed():
    """MSE 1.0 against scale 5/3 gives sqrt(0.6) = 0.774596..."""
    actual = np.array([[2.0, 2.0]])
    forecast = np.array([[1.0, 3.0]])
    scale = np.array([5 / 3])
    assert M.rmsse(actual, forecast, scale)[0] == pytest.approx(np.sqrt(0.6))


def test_rmsse_is_scale_free():
    """Doubling a series doubles its errors AND its scale, so RMSSE is fixed.

    This is the whole reason RMSSE exists: it lets a SKU selling 200 units a
    day and one selling 0.3 contribute comparably to an aggregate score.
    """
    small = np.array([[1.0, 3.0, 2.0, 5.0]])
    big = small * 100.0

    def score(series):
        hist, actual = series[:, :3], series[:, 3:]
        forecast = np.repeat(hist[:, -1:], actual.shape[1], axis=1)
        return M.rmsse(actual, forecast, M.naive_scale(hist))[0]

    assert score(small) == pytest.approx(score(big))


def test_rmsse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        M.rmsse(np.zeros((2, 3)), np.zeros((2, 4)), np.ones(2))


# --------------------------------------------------------------------------
# WRMSSE
# --------------------------------------------------------------------------
def test_series_weights_sum_to_one():
    w = M.series_weights(np.array([10.0, 30.0, 60.0]))
    assert w.sum() == pytest.approx(1.0)
    assert w[2] == pytest.approx(0.6)


def test_wrmsse_is_the_weighted_mean_of_rmsse():
    actual = np.array([[2.0, 2.0], [4.0, 4.0]])
    forecast = np.array([[1.0, 3.0], [4.0, 4.0]])
    scale = np.array([5 / 3, 2.0])
    weights = np.array([0.25, 0.75])

    per = M.rmsse(actual, forecast, scale)
    expected = 0.25 * per[0] + 0.75 * per[1]
    assert M.wrmsse(actual, forecast, scale, weights) == pytest.approx(expected)


def test_wrmsse_renormalises_over_scorable_series():
    """A series with a NaN scale drops out and the rest are re-weighted."""
    actual = np.array([[2.0, 2.0], [4.0, 4.0]])
    forecast = np.array([[1.0, 3.0], [4.0, 4.0]])
    scale = np.array([5 / 3, np.nan])
    weights = np.array([0.25, 0.75])

    per = M.rmsse(actual, forecast, scale)
    assert M.wrmsse(actual, forecast, scale, weights) == pytest.approx(per[0])


# --------------------------------------------------------------------------
# Pinball loss
# --------------------------------------------------------------------------
def test_pinball_penalises_asymmetrically():
    """At q = 0.9, missing low costs nine times missing high by the same amount."""
    under = M.pinball_loss(np.array([10.0]), np.array([8.0]), q=0.9)
    over = M.pinball_loss(np.array([8.0]), np.array([10.0]), q=0.9)

    assert under == pytest.approx(0.9 * 2)
    assert over == pytest.approx(0.1 * 2)
    assert under == pytest.approx(9 * over)


def test_pinball_is_symmetric_at_the_median():
    under = M.pinball_loss(np.array([10.0]), np.array([8.0]), q=0.5)
    over = M.pinball_loss(np.array([8.0]), np.array([10.0]), q=0.5)
    assert under == pytest.approx(over)


def test_pinball_minimiser_is_the_quantile():
    """The property the whole quantile-forecasting approach rests on.

    Sweep candidate forecasts over a known sample; the one minimising pinball
    loss at level q should be the empirical q-quantile.
    """
    rng = np.random.default_rng(0)
    sample = rng.poisson(3.0, size=4000).astype(float)

    for q in (0.5, 0.9):
        grid = np.arange(0.0, 15.0, 0.25)
        losses = [M.pinball_loss(sample, np.full_like(sample, g), q) for g in grid]
        best = grid[int(np.argmin(losses))]
        assert best == pytest.approx(np.quantile(sample, q), abs=0.5)


def test_pinball_rejects_bad_levels():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"q must be in"):
            M.pinball_loss(np.array([1.0]), np.array([1.0]), q=bad)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def test_coverage_counts_actuals_at_or_below_forecast():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    forecast = np.array([2.0, 2.0, 2.0, 2.0])
    assert M.coverage(actual, forecast) == pytest.approx(0.5)


def test_bias_sign_is_positive_when_over_forecasting():
    assert M.bias(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)
