"""LightGBM demand models: one point forecast, one model per quantile.

Two things here are deliberate and worth defending.

**Separate models per quantile, not one model with a spread assumption.**
An alternative is to forecast the mean and add a safety factor from an assumed
distribution (usually normal). That fails here: demand is zero on 55% of
SKU-days, so it is neither normal nor even continuous, and a normal-based
safety stock is wrong in a direction that costs money. Training directly on
pinball loss at each level estimates the quantile the inventory model needs
without assuming any distributional shape at all.

**Trained once, used for the whole 28-day horizon.** Because every feature in
`features.py` is lagged by at least the horizon length, a single model can
predict all 28 days directly. The usual alternative, recursive forecasting,
feeds each prediction back in as an input, which compounds error over the
horizon and makes honest validation awkward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:                                  # pragma: no cover
    raise ImportError("LightGBM is required: pip install lightgbm") from exc

from . import config as C
from .features import CATEGORICAL


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
BASE_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.06,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 2,
    "verbosity": -1,
    "seed": C.SEED,
}

# Tweedie is the standard choice for M5 point forecasts: it is designed for
# non-negative data with a point mass at zero, which describes this target
# exactly. Plain squared error would treat the zeros as ordinary observations
# and pull every prediction toward the middle of a distribution that has no
# middle.

N_ROUNDS = 400
EARLY_STOPPING = 30

# Training window. With two cores, using the full history costs a great deal
# of time for very little accuracy: the oldest years contribute least and the
# model already carries long lags. Two years is the compromise.
TRAIN_FROM_DAY = 1200


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def make_datasets(feats: pd.DataFrame, cols: list[str], valid_start: int,
                  valid_end: int, train_from: int = TRAIN_FROM_DAY):
    """Build LightGBM datasets with a strict time boundary between them."""
    train_mask = (feats["day_num"] >= train_from) & (feats["day_num"] < valid_start)
    valid_mask = (feats["day_num"] >= valid_start) & (feats["day_num"] < valid_end)

    train = feats.loc[train_mask]
    valid = feats.loc[valid_mask]

    if train.empty or valid.empty:
        raise ValueError("empty train or validation split; check the day bounds")
    if train["day_num"].max() >= valid["day_num"].min():
        raise ValueError("train window overlaps validation window")

    cat = [c for c in CATEGORICAL if c in cols]
    dtrain = lgb.Dataset(train[cols], label=train["sales"],
                         categorical_feature=cat, free_raw_data=False)
    dvalid = lgb.Dataset(valid[cols], label=valid["sales"],
                         categorical_feature=cat, reference=dtrain,
                         free_raw_data=False)
    return dtrain, dvalid, train, valid


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_point(dtrain, dvalid, verbose: bool = False):
    """Point forecast model, Tweedie objective."""
    return lgb.train(
        BASE_PARAMS,
        dtrain,
        num_boost_round=N_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=verbose),
            lgb.log_evaluation(50 if verbose else 0),
        ],
    )


def train_quantile(dtrain, dvalid, q: float, verbose: bool = False):
    """One quantile model, trained directly on pinball loss at level q."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")

    params = dict(BASE_PARAMS)
    params.update({
        "objective": "quantile",
        "alpha": q,
        "metric": "quantile",
    })
    return lgb.train(
        params,
        dtrain,
        num_boost_round=N_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=verbose),
            lgb.log_evaluation(50 if verbose else 0),
        ],
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict(model, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Predict, clipped at zero. Negative demand is not a thing."""
    raw = model.predict(frame[cols], num_iteration=model.best_iteration)
    return np.clip(raw, 0.0, None)


def to_matrix(frame: pd.DataFrame, values: np.ndarray, items: np.ndarray,
              days: np.ndarray) -> np.ndarray:
    """Reshape flat predictions into the (n_series, n_days) matrix layout."""
    out = frame[["item_id", "day_num"]].copy()
    out["value"] = values
    wide = out.pivot_table(index="item_id", columns="day_num",
                           values="value", aggfunc="mean", observed=True)
    wide = wide.reindex(index=items, columns=days)
    return np.nan_to_num(wide.to_numpy(dtype=float))


def importance(model, cols: list[str], top: int = 20) -> pd.DataFrame:
    """Gain-based feature importance, normalised to shares."""
    gain = model.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": cols, "gain": gain})
    df["share"] = df["gain"] / df["gain"].sum()
    return df.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
