"""Forecast accuracy metrics.

Two families are needed, for two different jobs.

Point forecasts are scored with RMSSE and WRMSSE, the official M5 metrics.
Plain RMSE cannot be averaged across SKUs: a SKU selling 200 units a day
would dominate a SKU selling 0.3, and the average would say nothing about
whether the model is any good on the long tail. RMSSE fixes this by dividing
each SKU's error by that SKU's own in-sample naive error, which puts every
series on the same scale. A value of 1.0 means "no better than predicting
yesterday's sales"; below 1.0 means better.

Quantile forecasts are scored with pinball loss, which is the loss function
whose minimiser *is* the quantile. That is not a convenience, it is the whole
reason quantile regression works, and it is the link to the inventory model
in Stage 4: the newsvendor wants a particular quantile, and pinball loss is
how we both train for it and check we got it.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Scaling denominator
# --------------------------------------------------------------------------
def naive_scale(train: np.ndarray, valid_from: np.ndarray | None = None) -> np.ndarray:
    """Mean squared one-step naive error, per series. The RMSSE denominator.

    For series i with training observations Y_1 ... Y_n:

        scale_i = (1 / (n - 1)) * sum_{t=2}^{n} (Y_t - Y_{t-1})^2

    This is the in-sample MSE of the random-walk forecast. Dividing by it
    makes the metric scale-free and unit-free.

    Parameters
    ----------
    train
        (n_series, n_days) array of training observations.
    valid_from
        Optional (n_series,) index of each series' first real observation.
        Days before it are ignored, so a SKU that launched late is scored on
        its actual history rather than on a run of structural zeros.

    Returns
    -------
    (n_series,) array. Series with fewer than two usable observations, or a
    perfectly flat history, get NaN and are excluded from the aggregate.
    """
    train = np.asarray(train, dtype=float)
    n_series, n_days = train.shape

    if valid_from is None:
        valid_from = np.zeros(n_series, dtype=int)
    valid_from = np.asarray(valid_from, dtype=int)

    cols = np.arange(n_days)
    mask = cols[None, :] >= valid_from[:, None]          # True where the series is live

    diff = np.diff(train, axis=1)
    # A difference is usable only when BOTH of its endpoints are live.
    diff_mask = mask[:, 1:] & mask[:, :-1]

    sq = np.where(diff_mask, diff ** 2, 0.0)
    counts = diff_mask.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = sq.sum(axis=1) / counts

    scale[counts < 1] = np.nan
    scale[scale == 0] = np.nan      # a flat series has no naive error to scale by
    return scale


# --------------------------------------------------------------------------
# Point-forecast metrics
# --------------------------------------------------------------------------
def rmsse(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Root mean squared scaled error, per series.

        RMSSE_i = sqrt( mean_h (Y - Yhat)^2  /  scale_i )
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    scale = np.asarray(scale, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape}, forecast {forecast.shape}")
    if scale.shape[0] != actual.shape[0]:
        raise ValueError("scale must have one entry per series")

    mse = np.mean((actual - forecast) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(mse / scale)


def wrmsse(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray,
           weights: np.ndarray) -> float:
    """Weighted RMSSE: the single number M5 ranked competitors on.

    Weights are each series' share of dollar sales in the last 28 days of
    training, so the score is dominated by the SKUs the business actually
    cares about rather than by the long tail of near-dead items.
    """
    per_series = rmsse(actual, forecast, scale)
    weights = np.asarray(weights, dtype=float)

    ok = np.isfinite(per_series) & np.isfinite(weights)
    if not ok.any():
        return float("nan")

    w = weights[ok]
    w = w / w.sum()                      # renormalise over the scorable series
    return float(np.sum(w * per_series[ok]))


def series_weights(revenue_last_28: np.ndarray) -> np.ndarray:
    """Normalised dollar-sales weights for WRMSSE."""
    rev = np.asarray(revenue_last_28, dtype=float)
    rev = np.where(np.isfinite(rev) & (rev > 0), rev, 0.0)
    total = rev.sum()
    if total <= 0:
        raise ValueError("no positive revenue; cannot build weights")
    return rev / total


# --------------------------------------------------------------------------
# Quantile metrics
# --------------------------------------------------------------------------
def pinball_loss(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss at level q.

        L = q * (y - yhat)        when y >= yhat   (under-forecast)
          = (1 - q) * (yhat - y)  when y <  yhat   (over-forecast)

    At q = 0.9 an under-forecast costs nine times an over-forecast of the
    same size, so the loss is minimised by a forecast that sits above the
    actual 90% of the time. That is exactly the 90th percentile, which is
    why minimising this loss estimates a quantile rather than a mean.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    diff = actual - forecast
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def mean_pinball(actual: np.ndarray, forecasts: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a set of quantile levels."""
    if not forecasts:
        raise ValueError("no quantile forecasts supplied")
    return float(np.mean([pinball_loss(actual, f, q) for q, f in forecasts.items()]))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def coverage(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Share of actuals at or below the forecast.

    For a well-calibrated quantile forecast at level q this should come out
    near q. It is the honesty check on Stage 3: a model can post a good
    pinball loss while being badly calibrated on a subset of series, and
    inventory decisions care about calibration, not average loss.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(actual <= forecast))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean forecast minus mean actual. Positive means over-forecasting."""
    return float(np.mean(forecast) - np.mean(actual))
