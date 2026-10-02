"""Baseline forecasters.

Every one of these is cheap, old and well understood. That is the point: a
gradient booster that cannot beat them has not earned its place, and without
them an accuracy number means nothing at all.

Two of these are specialists. Croston's method and the Syntetos-Boylan
Approximation were designed for exactly the intermittent demand that
dominates this dataset, and on the sparsest SKUs they are genuinely hard to
beat. If the model in Stage 3 loses to SBA on the intermittent segment, that
is a real finding about the data, not a bug.

Every function takes a (n_series, n_train_days) history and returns a
(n_series, horizon) forecast. Nothing here ever sees the evaluation window.
"""
from __future__ import annotations

import numpy as np

# Weekly seasonality. M5 is daily retail data; the dominant cycle is 7 days.
WEEK = 7


def _check(history: np.ndarray) -> np.ndarray:
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2:
        raise ValueError(f"history must be 2-D (n_series, n_days), got {hist.shape}")
    if hist.shape[1] < 1:
        raise ValueError("history has no observations")
    return hist


# --------------------------------------------------------------------------
# Simple baselines
# --------------------------------------------------------------------------
def zero(history: np.ndarray, horizon: int) -> np.ndarray:
    """Always forecast zero.

    Not a joke. When a SKU sells nothing on 58% of days, zero is the modal
    outcome, and on squared-error metrics this baseline is unpleasantly
    competitive on the sparsest series. It is here to keep everyone honest.
    """
    hist = _check(history)
    return np.zeros((hist.shape[0], horizon), dtype=float)


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value. The random walk."""
    hist = _check(history)
    return np.repeat(hist[:, -1:], horizon, axis=1)


def seasonal_naive(history: np.ndarray, horizon: int, period: int = WEEK) -> np.ndarray:
    """Repeat the value from `period` days ago, cycling through the horizon.

    Day 1 of the horizon takes last week's same weekday, and so on. Because
    the horizon (28) is longer than the period (7), the last week of history
    is tiled four times.
    """
    hist = _check(history)
    if hist.shape[1] < period:
        return naive(hist, horizon)

    last_cycle = hist[:, -period:]                       # (n_series, period)
    idx = np.arange(horizon) % period
    return last_cycle[:, idx]


def moving_average(history: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    """Flat forecast at the mean of the last `window` days."""
    hist = _check(history)
    window = min(window, hist.shape[1])
    level = hist[:, -window:].mean(axis=1, keepdims=True)
    return np.repeat(level, horizon, axis=1)


def seasonal_moving_average(history: np.ndarray, horizon: int,
                            period: int = WEEK, n_cycles: int = 8) -> np.ndarray:
    """Per-weekday mean over the last `n_cycles` weeks.

    A middle ground between the moving average (captures level, ignores the
    weekly cycle) and the seasonal naive (captures the cycle, but from a
    single noisy week).
    """
    hist = _check(history)
    take = min(period * n_cycles, hist.shape[1])
    recent = hist[:, -take:]

    # Align so that column j of `recent` maps to the right horizon position.
    offset = recent.shape[1] % period
    profile = np.zeros((hist.shape[0], period), dtype=float)
    for phase in range(period):
        cols = np.arange(recent.shape[1])
        sel = cols[(cols - offset) % period == phase]
        profile[:, phase] = recent[:, sel].mean(axis=1)

    idx = np.arange(horizon) % period
    return profile[:, idx]


# --------------------------------------------------------------------------
# Intermittent-demand specialists
# --------------------------------------------------------------------------
def _croston_state(history: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Run Croston's recursion and return the final (size, interval) estimates.

    Croston splits an intermittent series into two ordinary series and smooths
    each separately:

        z = exponentially smoothed size of non-zero demands
        p = exponentially smoothed interval between non-zero demands

    On a day with demand:      z <- a*y + (1-a)*z ;  p <- a*q + (1-a)*p ;  q <- 1
    On a day without demand:   estimates unchanged ;                       q <- q + 1

    The forecast is then z / p: an average demand per period that does not
    get dragged toward zero by the gaps, the way a plain moving average does.

    Vectorised across series, so the loop runs once per day rather than once
    per series-day.
    """
    hist = _check(history)
    n_series, n_days = hist.shape

    nonzero = hist > 0
    has_any = nonzero.any(axis=1)

    # Initialise from each series' first non-zero observation.
    first_idx = np.where(has_any, np.argmax(nonzero, axis=1), 0)
    rows = np.arange(n_series)

    z = np.where(has_any, hist[rows, first_idx], 0.0).astype(float)
    p = np.ones(n_series, dtype=float)
    q = np.ones(n_series, dtype=float)

    # Walk forward from the day after each series' first demand.
    start = int(first_idx.min()) + 1
    for t in range(start, n_days):
        live = has_any & (t > first_idx)
        if not live.any():
            continue

        y = hist[:, t]
        occurs = live & (y > 0)
        gap = live & (y <= 0)

        z[occurs] = alpha * y[occurs] + (1.0 - alpha) * z[occurs]
        p[occurs] = alpha * q[occurs] + (1.0 - alpha) * p[occurs]
        q[occurs] = 1.0
        q[gap] += 1.0

    p = np.maximum(p, 1.0)      # an interval below one period is not meaningful
    z = np.where(has_any, z, 0.0)
    return z, p


def croston(history: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Croston's method. Flat forecast at z / p."""
    z, p = _croston_state(history, alpha)
    level = (z / p)[:, None]
    return np.repeat(level, horizon, axis=1)


def sba(history: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan Approximation: Croston with the bias correction.

    Croston's estimator is biased upward, because the expectation of z/p is
    not the ratio of the expectations. Syntetos and Boylan showed the bias is
    approximately a factor of (1 - alpha/2), so SBA simply deflates Croston
    by that factor. It is usually the better of the two, and it is the
    standard benchmark for intermittent demand.
    """
    z, p = _croston_state(history, alpha)
    level = ((1.0 - alpha / 2.0) * z / p)[:, None]
    return np.repeat(level, horizon, axis=1)


# --------------------------------------------------------------------------
# Quantile baseline
# --------------------------------------------------------------------------
def empirical_quantile(history: np.ndarray, horizon: int, q: float,
                       window: int = 56) -> np.ndarray:
    """Flat forecast at the empirical q-quantile of the last `window` days.

    This is the baseline the Stage 3 quantile models must beat, and the
    simplest thing an inventory planner could actually do: look at recent
    demand, take the 90th percentile, stock that.
    """
    hist = _check(history)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")

    window = min(window, hist.shape[1])
    level = np.quantile(hist[:, -window:], q, axis=1, keepdims=True)
    return np.repeat(level, horizon, axis=1)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
POINT_BASELINES = {
    "zero": zero,
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average_28": moving_average,
    "seasonal_ma_8w": seasonal_moving_average,
    "croston": croston,
    "sba": sba,
}
