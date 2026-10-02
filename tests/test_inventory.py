"""Tests for the inventory model and simulator.

The newsvendor tests check the result against a brute-force expected-cost
minimisation rather than against a remembered number: if the critical-ratio
logic is wrong, a grid search over order quantities will disagree with it.
"""
import numpy as np
import pytest

from src import config as C
from src import inventory as INV


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
def test_costs_scale_with_price():
    costs = INV.build_cost_model(np.array([1.0, 10.0]))
    assert costs.unit_cost[1] == pytest.approx(10 * costs.unit_cost[0])
    assert costs.shortage[1] == pytest.approx(10 * costs.shortage[0])


def test_critical_ratio_varies_across_skus():
    """The bug this guards against shipped once and was invisible.

    With holding cost purely proportional to price, price cancels in the
    critical ratio and every SKU gets an identical one, silently turning a
    per-SKU newsvendor into a single global quantile. The flat shelf-space
    term is what prevents it, so a cheap SKU must end up with a lower
    critical ratio than an expensive one.
    """
    costs = INV.build_cost_model(np.array([0.5, 5.0, 50.0]))
    cr = costs.critical_ratio

    assert cr[0] < cr[1] < cr[2], "critical ratio must increase with price"
    assert cr.max() - cr.min() > 0.05, "critical ratio is effectively constant"
    assert np.all((cr > 0) & (cr < 1))


def test_critical_ratio_rises_with_the_stockout_penalty():
    cheap = INV.build_cost_model(np.array([5.0]), penalty_mult=0.5)
    dear = INV.build_cost_model(np.array([5.0]), penalty_mult=5.0)
    assert dear.critical_ratio[0] > cheap.critical_ratio[0]


# --------------------------------------------------------------------------
# The newsvendor result itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cu,co", [(9.0, 1.0), (1.0, 1.0), (1.0, 4.0)])
def test_newsvendor_quantile_minimises_expected_cost(cu, co):
    """Brute force: the critical-ratio quantile must beat every other order.

    Sample demand, sweep candidate order quantities, compute expected cost
    directly, and check the argmin lands on the quantile the critical ratio
    names. This is the formula Q* = F^-1(Cu / (Cu + Co)) verified rather than
    asserted.
    """
    rng = np.random.default_rng(11)
    demand = rng.poisson(6.0, size=20000).astype(float)

    cr = cu / (cu + co)
    predicted = np.quantile(demand, cr)

    grid = np.arange(0, 20, 1.0)
    costs = [(cu * np.maximum(demand - q, 0) + co * np.maximum(q - demand, 0)).mean()
             for q in grid]
    brute = grid[int(np.argmin(costs))]

    assert abs(brute - predicted) <= 1.0


def test_pick_quantile_level_snaps_to_the_nearest_available():
    cr = np.array([0.51, 0.88, 0.97])
    picked = INV.pick_quantile_level(cr, [0.5, 0.9, 0.95])
    np.testing.assert_allclose(picked, [0.5, 0.9, 0.95])


# --------------------------------------------------------------------------
# Order-up-to levels
# --------------------------------------------------------------------------
def test_quantile_level_is_never_below_the_mean_level():
    point = np.full((3, 20), 2.0)
    upper = point + 1.5
    t = 0
    mean_level = INV.order_up_to_mean(point, t)
    q_level = INV.order_up_to_quantile(point, upper, t)
    assert (q_level >= mean_level).all()


def test_mean_level_covers_exactly_the_protection_interval():
    point = np.full((1, 30), 3.0)
    level = INV.order_up_to_mean(point, 0, protection=10)
    assert level[0] == pytest.approx(30.0)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
def _flat_costs(n=1):
    return INV.CostModel(
        unit_cost=np.ones(n), holding_per_day=np.full(n, 0.1),
        shortage=np.full(n, 5.0), order_line=1.0,
    )


def test_perfect_foresight_and_huge_stock_means_no_shortage():
    demand = np.full((1, 14), 4.0)
    res = INV.simulate(demand, lambda t: np.array([1e6]), _flat_costs(),
                       lead_time=0, review=7)
    assert res.units_short == pytest.approx(0.0)
    assert res.fill_rate == pytest.approx(1.0)
    assert res.shortage_cost == pytest.approx(0.0)
    assert res.holding_cost > 0


def test_ordering_nothing_shorts_everything_after_the_opening_stock():
    demand = np.full((1, 10), 5.0)
    res = INV.simulate(demand, lambda t: np.array([0.0]), _flat_costs(),
                       lead_time=0, review=5, initial_days_cover=0.0)
    assert res.units_short == pytest.approx(demand.sum())
    assert res.fill_rate == pytest.approx(0.0)


def test_lead_time_delays_availability():
    """An order placed on day 0 with lead time 3 cannot serve day 0."""
    demand = np.zeros((1, 8))
    demand[0, 0] = 10.0
    res = INV.simulate(demand, lambda t: np.array([10.0]), _flat_costs(),
                       lead_time=3, review=8, initial_days_cover=0.0)
    assert res.units_short == pytest.approx(10.0)


def test_costs_sum_to_the_total():
    rng = np.random.default_rng(5)
    demand = rng.poisson(3.0, size=(6, 21)).astype(float)
    res = INV.simulate(demand, lambda t: np.full(6, 25.0), _flat_costs(6))
    assert res.total_cost == pytest.approx(
        res.holding_cost + res.shortage_cost + res.ordering_cost)
    assert res.per_sku_cost.sum() == pytest.approx(res.total_cost, rel=1e-9)


def test_fill_rate_is_bounded():
    rng = np.random.default_rng(6)
    demand = rng.poisson(2.0, size=(4, 28)).astype(float)
    for level in (0.0, 5.0, 500.0):
        res = INV.simulate(demand, lambda t, L=level: np.full(4, L), _flat_costs(4))
        assert 0.0 <= res.fill_rate <= 1.0
        assert 0.0 <= res.cycle_service <= 1.0


def test_a_higher_order_up_to_level_never_increases_shortage():
    """Monotonicity. More stock cannot cause more stockouts."""
    rng = np.random.default_rng(9)
    demand = rng.poisson(4.0, size=(5, 28)).astype(float)
    costs = _flat_costs(5)

    short = [INV.simulate(demand, lambda t, L=L: np.full(5, L), costs).units_short
             for L in (10.0, 30.0, 60.0, 120.0)]
    assert short == sorted(short, reverse=True)
