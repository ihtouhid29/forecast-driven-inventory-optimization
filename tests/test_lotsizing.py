"""Tests for the capacitated lot-sizing model.

The valuable tests here are the ones with a known optimum. A MIP that returns
"OPTIMAL" is telling you it solved the model you wrote, not that the model is
the one you meant, so each case below is small enough that the right answer
can be worked out by hand.
"""
import numpy as np
import pytest

from src import lotsizing as LS


# --------------------------------------------------------------------------
# Known optima
# --------------------------------------------------------------------------
def test_unconstrained_plan_meets_demand_exactly():
    """With ample capacity and positive holding cost, order just in time.

    Ordering early costs holding; ordering late costs shortage. With both
    positive and capacity slack, the optimum is to order each period's demand
    in that period and carry nothing.
    """
    demand = np.array([[5.0, 3.0, 7.0]])
    res = LS.solve_clsp(
        demand=demand,
        holding=np.array([1.0]),
        shortage_cost=np.array([100.0]),
        setup_cost=0.0,
        capacity=np.array([100.0, 100.0, 100.0]),
        time_limit=10.0,
    )
    assert res.proven_optimal
    np.testing.assert_allclose(res.orders, demand)
    assert res.inventory.sum() == pytest.approx(0.0)
    assert res.shortage.sum() == pytest.approx(0.0)


def test_high_setup_cost_forces_batching():
    """When opening a line costs more than carrying stock, orders consolidate.

    Three periods of demand 5. Holding is cheap (0.01/unit/period), a setup
    costs 50. Three separate orders cost 150 in setups; one order of 15 in
    period 0 costs 50 plus a little holding. The solver must find the batch.
    """
    demand = np.full((1, 3), 5.0)
    res = LS.solve_clsp(
        demand=demand,
        holding=np.array([0.01]),
        shortage_cost=np.array([100.0]),
        setup_cost=50.0,
        capacity=np.full(3, 100.0),
        time_limit=10.0,
    )
    assert res.proven_optimal
    assert res.setups.sum() == 1, f"expected one order, got {res.setups.sum()}"
    assert res.orders[0, 0] == pytest.approx(15.0)


def test_capacity_forces_early_purchase():
    """Demand that cannot be met in its own period must be bought earlier.

    Period 0 needs nothing, period 1 needs 10, but period 1 can only take 4.
    The only feasible way to serve it is to buy ahead in period 0.
    """
    demand = np.array([[0.0, 10.0]])
    res = LS.solve_clsp(
        demand=demand,
        holding=np.array([0.1]),
        shortage_cost=np.array([50.0]),
        setup_cost=0.0,
        capacity=np.array([20.0, 4.0]),
        time_limit=10.0,
    )
    assert res.proven_optimal
    assert res.orders[0, 0] >= 6.0
    assert res.shortage.sum() == pytest.approx(0.0)


def test_scarce_capacity_is_given_to_the_expensive_shortage():
    """With only enough for one SKU, serve the one whose stockout costs more."""
    demand = np.array([[10.0], [10.0]])
    res = LS.solve_clsp(
        demand=demand,
        holding=np.array([0.1, 0.1]),
        shortage_cost=np.array([1.0, 100.0]),
        setup_cost=0.0,
        capacity=np.array([10.0]),
        time_limit=10.0,
    )
    assert res.proven_optimal
    assert res.orders[1, 0] > res.orders[0, 0]
    assert res.orders[1, 0] == pytest.approx(10.0)


def test_opening_stock_reduces_the_order():
    demand = np.array([[8.0]])
    res = LS.solve_clsp(
        demand=demand, holding=np.array([0.1]), shortage_cost=np.array([50.0]),
        setup_cost=0.0, capacity=np.array([100.0]),
        opening_stock=np.array([3.0]), time_limit=10.0,
    )
    assert res.proven_optimal
    assert res.orders[0, 0] == pytest.approx(5.0)


def test_capacity_is_never_exceeded():
    rng = np.random.default_rng(4)
    demand = rng.poisson(6.0, size=(12, 4)).astype(float)
    cap = np.full(4, 30.0)
    res = LS.solve_clsp(
        demand=demand, holding=np.full(12, 0.2),
        shortage_cost=np.full(12, 4.0), setup_cost=1.0,
        capacity=cap, time_limit=20.0,
    )
    assert (res.orders.sum(axis=0) <= cap + 1e-6).all()


def test_rejects_mismatched_capacity_length():
    with pytest.raises(ValueError, match="one entry per period"):
        LS.solve_clsp(
            demand=np.ones((2, 3)), holding=np.ones(2),
            shortage_cost=np.ones(2), setup_cost=0.0,
            capacity=np.ones(2), time_limit=5.0,
        )


# --------------------------------------------------------------------------
# The heuristic
# --------------------------------------------------------------------------
def test_rationing_leaves_a_feasible_plan_untouched():
    target = np.array([[3.0, 4.0], [2.0, 1.0]])
    out = LS.proportional_rationing(target, np.array([100.0, 100.0]))
    np.testing.assert_allclose(out, target)


def test_rationing_scales_everyone_by_the_same_fraction():
    target = np.array([[60.0], [40.0]])
    out = LS.proportional_rationing(target, np.array([50.0]))
    assert out.sum() <= 50.0
    assert out[0, 0] / out[1, 0] == pytest.approx(1.5, rel=0.05)


def test_rationing_never_exceeds_capacity():
    rng = np.random.default_rng(2)
    target = rng.integers(0, 40, size=(20, 3)).astype(float)
    cap = np.array([50.0, 10.0, 200.0])
    out = LS.proportional_rationing(target, cap)
    assert (out.sum(axis=0) <= cap + 1e-9).all()


# --------------------------------------------------------------------------
# Shared evaluation
# --------------------------------------------------------------------------
def test_evaluate_plan_accounting_is_self_consistent():
    orders = np.array([[10.0, 0.0]])
    demand = np.array([[4.0, 4.0]])
    out = LS.evaluate_plan(orders, demand, np.array([1.0]), np.array([10.0]), 2.0)

    # Day 1: 10 in, 4 served, 6 held. Day 2: 4 served, 2 held. No shortage.
    assert out["units_short"] == pytest.approx(0.0)
    assert out["fill_rate"] == pytest.approx(1.0)
    assert out["holding_cost"] == pytest.approx(6.0 + 2.0)
    assert out["setup_cost"] == pytest.approx(2.0)
    assert out["total_cost"] == pytest.approx(out["holding_cost"]
                                              + out["shortage_cost"]
                                              + out["setup_cost"])


def test_evaluate_plan_charges_for_shortage():
    out = LS.evaluate_plan(np.zeros((1, 1)), np.array([[5.0]]),
                           np.array([1.0]), np.array([3.0]), 0.0)
    assert out["units_short"] == pytest.approx(5.0)
    assert out["shortage_cost"] == pytest.approx(15.0)
    assert out["fill_rate"] == pytest.approx(0.0)


def test_optimiser_beats_rationing_when_capacity_binds_and_costs_differ():
    """The claim Stage 5 rests on, on a case small enough to verify by hand.

    Two SKUs, equal demand, capacity for only half of it. One SKU's stockout
    costs 100x the other's. Rationing splits the capacity evenly; the
    optimiser gives it all to the expensive one. The optimiser must win.
    """
    demand = np.array([[10.0], [10.0]])
    holding = np.array([0.1, 0.1])
    shortage = np.array([1.0, 100.0])
    cap = np.array([10.0])

    heur = LS.proportional_rationing(demand, cap)
    opt = LS.solve_clsp(demand=demand, holding=holding, shortage_cost=shortage,
                        setup_cost=0.0, capacity=cap, time_limit=10.0).orders

    h = LS.evaluate_plan(heur, demand, holding, shortage, 0.0)
    o = LS.evaluate_plan(opt, demand, holding, shortage, 0.0)
    assert o["total_cost"] < h["total_cost"]
