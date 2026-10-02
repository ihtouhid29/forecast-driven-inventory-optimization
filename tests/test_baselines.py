"""Tests for the baseline forecasters.

The most important test in this file is `test_no_baseline_sees_the_future`.
Leakage is the failure mode that makes a forecasting project silently
worthless, and it is invisible in the output: scores just look good.
"""
import numpy as np
import pytest

from src import baselines as B

HORIZON = 5


def _toy() -> np.ndarray:
    """Two series, 14 days, with a deliberate weekly pattern in the first."""
    week = [5.0, 9.0, 1.0, 0.0, 0.0, 2.0, 3.0]
    return np.array([week * 2, [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0] * 2])


# --------------------------------------------------------------------------
# Shape and leakage contracts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn", sorted(B.POINT_BASELINES.items()))
def test_shape_contract(name, fn):
    out = fn(_toy(), HORIZON)
    assert out.shape == (2, HORIZON), f"{name} returned {out.shape}"
    assert np.isfinite(out).all(), f"{name} produced non-finite values"


@pytest.mark.parametrize("name,fn", sorted(B.POINT_BASELINES.items()))
def test_no_baseline_sees_the_future(name, fn):
    """A forecast must depend only on the history it was handed.

    We fit on the same history twice. If any baseline reached past the end of
    its input, appending different future data would change the answer. It
    cannot here, because the future is never passed in, and this test is what
    keeps that true as the code changes.
    """
    hist = _toy()
    first = fn(hist, HORIZON)
    second = fn(hist.copy(), HORIZON)
    np.testing.assert_allclose(first, second)

    # Mutating the caller's array afterwards must not alter a stored forecast.
    hist[:, -1] = 999.0
    np.testing.assert_allclose(first, fn(_toy(), HORIZON))


@pytest.mark.parametrize("name,fn", sorted(B.POINT_BASELINES.items()))
def test_rejects_one_dimensional_input(name, fn):
    with pytest.raises(ValueError, match="2-D"):
        fn(np.array([1.0, 2.0, 3.0]), HORIZON)


# --------------------------------------------------------------------------
# Individual methods
# --------------------------------------------------------------------------
def test_naive_repeats_the_last_value():
    out = B.naive(_toy(), HORIZON)
    assert (out[0] == 3.0).all()
    assert (out[1] == 0.0).all()


def test_seasonal_naive_tiles_the_last_week():
    """Horizon day 0 takes the value 7 days back, and the pattern repeats."""
    out = B.seasonal_naive(_toy(), horizon=9, period=7)
    expected_cycle = [5.0, 9.0, 1.0, 0.0, 0.0, 2.0, 3.0]
    np.testing.assert_allclose(out[0], expected_cycle + expected_cycle[:2])


def test_seasonal_naive_falls_back_when_history_is_short():
    hist = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        B.seasonal_naive(hist, HORIZON, period=7), B.naive(hist, HORIZON))


def test_moving_average_is_the_window_mean():
    hist = np.array([[1.0, 2.0, 3.0, 10.0]])
    out = B.moving_average(hist, HORIZON, window=3)
    assert out[0, 0] == pytest.approx(5.0)          # mean of 2, 3, 10
    assert np.allclose(out, out[0, 0])              # flat across the horizon


def test_zero_baseline_is_zero():
    assert (B.zero(_toy(), HORIZON) == 0).all()


# --------------------------------------------------------------------------
# Croston and SBA
# --------------------------------------------------------------------------
def _croston_reference(series: np.ndarray, alpha: float) -> float:
    """A deliberately naive scalar implementation of Croston's recursion.

    The production version is vectorised across thousands of series at once,
    which is exactly the kind of code that quietly gets an index wrong. This
    plain loop is written straight from the textbook definition and exists
    only to disagree with it if the fast version drifts.
    """
    nonzero = np.flatnonzero(series > 0)
    if nonzero.size == 0:
        return 0.0

    start = int(nonzero[0])
    z = float(series[start])
    p = 1.0
    q = 1.0

    for t in range(start + 1, len(series)):
        y = float(series[t])
        if y > 0:
            z = alpha * y + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1.0
        else:
            q += 1.0
    return z / max(p, 1.0)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_croston_matches_the_scalar_reference(alpha):
    rng = np.random.default_rng(7)
    series = (rng.random((6, 60)) < 0.25) * rng.integers(1, 9, size=(6, 60))
    series = series.astype(float)

    fast = B.croston(series, horizon=3, alpha=alpha)[:, 0]
    slow = np.array([_croston_reference(row, alpha) for row in series])
    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_sba_is_croston_deflated_by_the_bias_factor():
    """SBA = (1 - alpha/2) * Croston. The Syntetos-Boylan correction."""
    alpha = 0.1
    series = _toy()
    c = B.croston(series, HORIZON, alpha=alpha)
    s = B.sba(series, HORIZON, alpha=alpha)
    np.testing.assert_allclose(s, (1 - alpha / 2) * c, rtol=1e-12)
    assert (s <= c + 1e-12).all(), "SBA must not exceed Croston"


def test_croston_handles_an_all_zero_series():
    out = B.croston(np.zeros((1, 20)), HORIZON)
    assert (out == 0).all()


def test_croston_beats_moving_average_on_a_sparse_spiky_series():
    """The case Croston was designed for.

    Demand of 10 units every 10th day. The true mean rate is 1.0/day. Croston
    estimates size and interval separately and recovers it; it should land
    closer than a flat mean over a window that may or may not contain a spike.
    """
    series = np.zeros((1, 200))
    series[0, ::10] = 10.0

    c = B.croston(series, 1, alpha=0.1)[0, 0]
    assert c == pytest.approx(1.0, abs=0.25)


# --------------------------------------------------------------------------
# Quantile baseline
# --------------------------------------------------------------------------
def test_empirical_quantile_is_monotone_in_q():
    hist = np.random.default_rng(3).poisson(2.0, size=(4, 80)).astype(float)
    levels = [0.5, 0.75, 0.9, 0.99]
    vals = [B.empirical_quantile(hist, 1, q)[:, 0] for q in levels]
    for lo, hi in zip(vals, vals[1:]):
        assert (hi >= lo - 1e-12).all()


def test_empirical_quantile_rejects_bad_levels():
    with pytest.raises(ValueError, match=r"q must be in"):
        B.empirical_quantile(_toy(), HORIZON, q=1.0)
