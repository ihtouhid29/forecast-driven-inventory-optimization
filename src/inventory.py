"""Inventory policies and a simulator that charges them for their mistakes.

This is where the forecast stops being a number and becomes a decision.

**The newsvendor result.** Order Q units against uncertain demand D. Each unit
short costs Cu (lost margin, plus whatever a disappointed customer is worth).
Each unit left over costs Co (holding, capital, spoilage). Expected cost is
minimised where the probability of running short equals Co / (Cu + Co), which
means ordering the quantile:

    Q* = F^-1( Cu / (Cu + Co) )

The ratio Cu / (Cu + Co) is the *critical ratio*. It is the whole reason
Stage 3 trains quantile models: the optimal order is not the average demand,
it is a specific percentile of the demand distribution, chosen by the relative
cost of the two ways of being wrong. When shortage is expensive relative to
holding, the critical ratio is high and you deliberately over-stock.

**Periodic review.** Real ordering is not a single decision. Stock is reviewed
every R days and arrives L days later, so an order must cover demand over the
protection interval R + L, not over one day. The policy is order-up-to-S:
at each review, order the difference between S and the inventory position.

**Unmet demand is lost, not backordered.** Grocery retail: the customer buys
a different brand and does not come back for that unit. This matters, because
backordered demand would eventually be served and lost demand never is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Cost structure
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """Per-unit economics for one SKU, derived from its shelf price.

    M5 supplies prices but no cost of goods, no warehousing rate and no
    estimate of what a stockout costs. Everything below is an assumption,
    stated explicitly so it can be varied in the sensitivity analysis.
    """
    unit_cost: np.ndarray        # cost of goods per unit
    holding_per_day: np.ndarray  # carrying cost per unit per day
    shortage: np.ndarray         # cost per unit of unmet demand
    order_line: float            # fixed cost of placing one SKU line

    @property
    def critical_ratio(self) -> np.ndarray:
        """Cu / (Cu + Co), with Co taken over the protection interval.

        Co is the holding cost of carrying one surplus unit for the interval
        the order is meant to cover; Cu is the shortage cost. Both must be
        measured over the same period or the ratio is meaningless.
        """
        co = self.holding_per_day * C.PROTECTION_DAYS
        return self.shortage / (self.shortage + co)


def build_cost_model(prices: np.ndarray,
                     gross_margin: float = C.GROSS_MARGIN,
                     holding_rate: float = C.UNIT_HOLDING_COST_RATE,
                     penalty_mult: float = C.STOCKOUT_PENALTY_MULT,
                     order_line: float = C.ORDER_LINE_COST,
                     space_cost: float = C.SPACE_COST_PER_UNIT_DAY) -> CostModel:
    """Derive per-SKU costs from mean shelf price.

    Holding cost has two parts, and the second one is load-bearing:

        capital  = holding_rate * unit_cost / 365      (scales with price)
        space    = space_cost                          (flat per unit)

    A first version of this model had only the capital term. Because shortage
    cost is also proportional to price, price cancelled in the critical ratio
    and all 3,049 SKUs came out at exactly 0.9895 -- the per-SKU newsvendor
    was silently a single global quantile. The flat space term breaks that
    degeneracy and encodes something true: a cheap bulky item occupies the
    same shelf as an expensive one, so it is costlier to hold per dollar of
    value and should be allowed to stock out more often.
    """
    prices = np.asarray(prices, dtype=float)
    prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    prices = np.nan_to_num(prices, nan=float(np.nanmedian(prices)))

    unit_cost = prices * (1.0 - gross_margin)
    margin = prices * gross_margin

    return CostModel(
        unit_cost=unit_cost,
        holding_per_day=holding_rate * unit_cost / 365.0 + space_cost,
        shortage=penalty_mult * margin,
        order_line=order_line,
    )


# --------------------------------------------------------------------------
# Order-up-to levels
# --------------------------------------------------------------------------
def pick_quantile_level(critical_ratio: np.ndarray,
                        available: list[float]) -> np.ndarray:
    """Map each SKU's critical ratio to the nearest trained quantile level."""
    levels = np.asarray(sorted(available), dtype=float)
    idx = np.abs(levels[None, :] - critical_ratio[:, None]).argmin(axis=1)
    return levels[idx]


def order_up_to_mean(point_forecast: np.ndarray, t: int,
                     protection: int = C.PROTECTION_DAYS) -> np.ndarray:
    """Cover the expected demand over the protection interval, nothing more.

    Deliberately naive. It is here to show what a point forecast alone buys
    you: with no safety stock, a policy that targets the mean is short roughly
    half the time, and on intermittent demand rather more often than that.
    """
    window = point_forecast[:, t:t + protection]
    return window.sum(axis=1)


def order_up_to_quantile(point_forecast: np.ndarray,
                         quantile_forecast: np.ndarray, t: int,
                         protection: int = C.PROTECTION_DAYS) -> np.ndarray:
    """Newsvendor level: expected demand plus safety stock from the quantile.

    The daily quantile forecasts give a daily safety margin (quantile minus
    mean). Scaling it to the protection interval uses the square-root-of-time
    rule from inventory theory: for demand that is independent across days,
    the standard deviation of the sum grows with the square root of the
    number of days, not linearly.

    That independence assumption is the main simplification in this stage.
    Real demand is positively autocorrelated (a good week tends to follow a
    good week), so the true interval variance is larger and this rule
    under-provisions slightly. Summing the daily quantiles instead would err
    the other way, and by more: the sum of quantiles strictly exceeds the
    quantile of the sum, so it over-stocks. The square-root rule is the
    standard compromise and the closer of the two.
    """
    mean_window = point_forecast[:, t:t + protection]
    q_window = quantile_forecast[:, t:t + protection]

    expected = mean_window.sum(axis=1)
    daily_safety = np.maximum(q_window - mean_window, 0.0).mean(axis=1)
    return expected + daily_safety * np.sqrt(protection)


def order_up_to_historical(history: np.ndarray, critical_ratio: np.ndarray,
                           protection: int = C.PROTECTION_DAYS,
                           window: int = 56) -> np.ndarray:
    """The no-forecast baseline: a textbook reorder point from recent history.

    Mean and standard deviation of the last `window` days, scaled to the
    protection interval, with a safety factor read off the normal
    distribution at the critical ratio. This is what a planner does with a
    spreadsheet and no model, and it is the policy the forecast has to beat.

    The normal assumption is wrong here in an interesting way, and that is
    part of the point: demand is intermittent and right-skewed, so a normal
    safety factor misprices the tail. Stage 6 reports what that costs.
    """
    from scipy.stats import norm

    recent = history[:, -window:]
    mu = recent.mean(axis=1)
    sigma = recent.std(axis=1, ddof=1)

    z = norm.ppf(np.clip(critical_ratio, 1e-6, 1 - 1e-6))
    return mu * protection + z * sigma * np.sqrt(protection)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    total_cost: float
    holding_cost: float
    shortage_cost: float
    ordering_cost: float
    fill_rate: float
    cycle_service: float
    mean_on_hand: float
    units_short: float
    units_demanded: float
    orders_placed: float
    per_sku_cost: np.ndarray


def simulate(demand: np.ndarray, order_up_to, costs: CostModel,
             lead_time: int = C.LEAD_TIME_DAYS,
             review: int = C.REVIEW_PERIOD_DAYS,
             initial_days_cover: float = 1.0) -> SimResult:
    """Run a periodic-review, order-up-to policy against realised demand.

    Parameters
    ----------
    demand
        (n_series, n_days) of what actually happened. The policy never sees
        this; only the simulator does, when it fills orders.
    order_up_to
        Callable taking the review day index and returning the (n_series,)
        order-up-to level for that review.
    initial_days_cover
        Opening stock, expressed in days of mean demand. Every policy starts
        from the same position so the comparison is not decided by who was
        handed more inventory on day one.

    Timing within a day: order arrivals land first, then demand is served
    from whatever is on hand, then holding is charged on the closing balance.
    """
    demand = np.asarray(demand, dtype=float)
    n_series, n_days = demand.shape

    mean_daily = demand.mean(axis=1)
    on_hand = mean_daily * initial_days_cover
    pipeline = np.zeros((n_series, n_days + lead_time + 1), dtype=float)

    holding = shortage = ordering = 0.0
    units_short = np.zeros(n_series)
    per_sku_cost = np.zeros(n_series)
    stockout_days = 0
    orders_placed = 0.0

    for t in range(n_days):
        # Order first, then receive. Doing it the other way round silently
        # dropped same-day arrivals when lead_time was 0: the order landed in
        # a pipeline slot that had already been swept into on_hand for that
        # day. `test_perfect_foresight_and_huge_stock_means_no_shortage`
        # caught it, reporting 52 units short against an order-up-to level of
        # one million. The inventory position below therefore counts the
        # pipeline from `t` onward, since today's arrival has not landed yet.
        if t % review == 0:
            level = np.asarray(order_up_to(t), dtype=float)
            position = on_hand + pipeline[:, t:].sum(axis=1)
            qty = np.maximum(np.round(level - position), 0.0)

            arrival = min(t + lead_time, n_days + lead_time)
            pipeline[:, arrival] += qty

            placed = qty > 0
            cost = placed.sum() * costs.order_line
            ordering += cost
            per_sku_cost += placed * costs.order_line
            orders_placed += placed.sum()

        on_hand += pipeline[:, t]

        served = np.minimum(on_hand, demand[:, t])
        short = demand[:, t] - served
        on_hand -= served

        s_cost = short * costs.shortage
        h_cost = on_hand * costs.holding_per_day

        shortage += s_cost.sum()
        holding += h_cost.sum()
        per_sku_cost += s_cost + h_cost
        units_short += short
        stockout_days += int((short > 0).sum())

    demanded = demand.sum()
    total = holding + shortage + ordering

    return SimResult(
        total_cost=float(total),
        holding_cost=float(holding),
        shortage_cost=float(shortage),
        ordering_cost=float(ordering),
        fill_rate=float(1.0 - units_short.sum() / demanded) if demanded else 1.0,
        cycle_service=float(1.0 - stockout_days / (n_series * n_days)),
        mean_on_hand=float(np.mean(on_hand)),
        units_short=float(units_short.sum()),
        units_demanded=float(demanded),
        orders_placed=float(orders_placed),
        per_sku_cost=per_sku_cost,
    )
