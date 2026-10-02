"""Capacitated lot-sizing, solved with CP-SAT.

Stage 4 sets each SKU's order independently. That is correct only while the
warehouse has room for whatever every SKU independently asks for. Once a
shared constraint binds -- a delivery slot, a truck, a rack -- the SKUs are
competing, and no per-SKU rule can resolve the competition well, because the
right answer for one SKU depends on what the others are doing.

That is the capacitated lot-sizing problem (CLSP), and it is NP-hard. Written
out, for items i over periods t:

    minimise   sum_i sum_t [ h_i * I_it  +  p_i * s_it  +  K * y_it ]

    subject to  I_it = I_i,t-1 + x_it - d_it + s_it     (inventory balance)
                sum_i v_i * x_it  <=  Cap_t              (shared capacity)
                x_it <= M_i * y_it                       (setup forcing)
                x_it, I_it, s_it >= 0,   y_it in {0,1}

`x` is how much to order, `I` what is left over, `s` what could not be served,
`y` a binary that is 1 whenever a SKU is ordered at all, so the fixed cost K
of putting a line on an order is charged once per line and not per unit. The
setup-forcing constraint is what makes this integer programming rather than a
linear program, and it is what pushes the solution toward batching: if you
are paying K anyway, order enough to skip next period.

**Why CP-SAT rather than an LP solver.** The binaries make this a mixed
integer program, so a pure LP solver cannot express it. CP-SAT handles the
integrality directly, is genuinely good at this structure, ships in OR-Tools
under an open licence, and returns a proven optimality gap rather than just
an answer -- which matters, because a heuristic that cannot tell you how far
from optimal it is cannot be compared against anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ortools.sat.python import cp_model
except ImportError as exc:                                  # pragma: no cover
    raise ImportError("OR-Tools is required: pip install ortools") from exc


# CP-SAT is an integer solver. Costs are scaled to integer cents-equivalents
# and quantities are kept in whole units, which is what ordering actually is.
COST_SCALE = 1000


@dataclass
class LotSizingResult:
    orders: np.ndarray          # (n_items, n_periods) units to order
    inventory: np.ndarray       # (n_items, n_periods) closing inventory
    shortage: np.ndarray        # (n_items, n_periods) unmet demand
    setups: np.ndarray          # (n_items, n_periods) 0/1 order lines opened
    objective: float            # total cost in dollars
    best_bound: float           # proven lower bound in dollars
    gap: float                  # relative optimality gap
    status: str
    wall_time: float

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"


def solve_clsp(demand: np.ndarray,
               holding: np.ndarray,
               shortage_cost: np.ndarray,
               setup_cost: float,
               capacity: np.ndarray,
               volume: np.ndarray | None = None,
               opening_stock: np.ndarray | None = None,
               time_limit: float = 60.0,
               workers: int = 2) -> LotSizingResult:
    """Solve the capacitated lot-sizing problem.

    Parameters
    ----------
    demand
        (n_items, n_periods) demand to plan against. This is where the
        forecast enters: pass the quantile forecast aggregated to periods
        and the optimiser is planning against the same distributional view
        the newsvendor uses.
    holding, shortage_cost
        (n_items,) per-unit-per-period costs.
    setup_cost
        Cost of opening one order line in one period.
    capacity
        (n_periods,) shared capacity per period, in volume units.
    volume
        (n_items,) space each unit consumes. Defaults to 1 unit each.
    opening_stock
        (n_items,) stock on hand before period 0.
    """
    demand = np.asarray(demand, dtype=float)
    n_items, n_periods = demand.shape

    volume = np.ones(n_items) if volume is None else np.asarray(volume, dtype=float)
    opening = np.zeros(n_items) if opening_stock is None else np.asarray(opening_stock, dtype=float)

    if capacity.shape[0] != n_periods:
        raise ValueError("capacity must have one entry per period")

    dem = np.maximum(np.round(demand), 0).astype(int)
    cap = np.maximum(np.round(capacity), 0).astype(int)
    vol = np.maximum(np.round(volume), 1).astype(int)
    open_stock = np.maximum(np.round(opening), 0).astype(int)

    h = np.round(holding * COST_SCALE).astype(int)
    p = np.round(shortage_cost * COST_SCALE).astype(int)
    k = int(round(setup_cost * COST_SCALE))

    # An order never needs to exceed all remaining demand for that SKU.
    max_order = np.maximum(dem.sum(axis=1), 1)

    model = cp_model.CpModel()

    x, inv, short, setup = {}, {}, {}, {}
    for i in range(n_items):
        ub = int(max_order[i])
        for t in range(n_periods):
            x[i, t] = model.NewIntVar(0, ub, f"x_{i}_{t}")
            inv[i, t] = model.NewIntVar(0, ub, f"I_{i}_{t}")
            short[i, t] = model.NewIntVar(0, int(dem[i, t]), f"s_{i}_{t}")
            setup[i, t] = model.NewBoolVar(f"y_{i}_{t}")

            # Inventory balance. Shortage is lost, not carried forward.
            prev = inv[i, t - 1] if t > 0 else int(open_stock[i])
            model.Add(inv[i, t] == prev + x[i, t] - int(dem[i, t]) + short[i, t])

            # Setup forcing: ordering anything opens the line.
            model.Add(x[i, t] <= ub * setup[i, t])

    for t in range(n_periods):
        model.Add(sum(int(vol[i]) * x[i, t] for i in range(n_items)) <= int(cap[t]))

    model.Minimize(
        sum(int(h[i]) * inv[i, t] + int(p[i]) * short[i, t] + k * setup[i, t]
            for i in range(n_items) for t in range(n_periods))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no feasible plan found: {status_name}")

    def grab(store):
        return np.array([[solver.Value(store[i, t]) for t in range(n_periods)]
                         for i in range(n_items)], dtype=float)

    obj = solver.ObjectiveValue() / COST_SCALE
    bound = solver.BestObjectiveBound() / COST_SCALE
    gap = abs(obj - bound) / max(abs(obj), 1e-9)

    return LotSizingResult(
        orders=grab(x), inventory=grab(inv), shortage=grab(short),
        setups=grab(setup), objective=float(obj), best_bound=float(bound),
        gap=float(gap), status=status_name, wall_time=float(solver.WallTime()),
    )


# --------------------------------------------------------------------------
# The heuristic it has to beat
# --------------------------------------------------------------------------
def proportional_rationing(target: np.ndarray, capacity: np.ndarray,
                           volume: np.ndarray | None = None) -> np.ndarray:
    """What a planner does without an optimiser: scale everyone down equally.

    Each SKU asks for its independently-computed newsvendor quantity. If the
    total exceeds capacity, every request is cut by the same proportion.

    It is fair, it is what most spreadsheets do, and it is wrong: it takes the
    same fraction from a SKU whose stockout costs pennies as from one whose
    stockout costs dollars, and it never shifts a purchase into an adjacent
    period where there is room. Those are exactly the two moves the optimiser
    makes, and the gap between them is what Stage 5 measures.
    """
    target = np.asarray(target, dtype=float)
    n_items, n_periods = target.shape
    volume = np.ones(n_items) if volume is None else np.asarray(volume, dtype=float)

    out = target.copy()
    for t in range(n_periods):
        used = float((volume * out[:, t]).sum())
        if used > capacity[t] and used > 0:
            out[:, t] *= capacity[t] / used
    return np.floor(out)


def evaluate_plan(orders: np.ndarray, demand: np.ndarray,
                  holding: np.ndarray, shortage_cost: np.ndarray,
                  setup_cost: float,
                  opening_stock: np.ndarray | None = None) -> dict:
    """Cost any order plan against realised demand, on identical accounting.

    Used to score the optimiser and the heuristic the same way, so the
    comparison cannot be won by measuring them differently.
    """
    orders = np.asarray(orders, dtype=float)
    demand = np.asarray(demand, dtype=float)
    n_items, n_periods = demand.shape

    inv = np.zeros(n_items) if opening_stock is None else np.array(opening_stock, dtype=float)
    holding_total = shortage_total = 0.0
    units_short = 0.0

    for t in range(n_periods):
        inv = inv + orders[:, t]
        served = np.minimum(inv, demand[:, t])
        short = demand[:, t] - served
        inv = inv - served

        holding_total += float((holding * inv).sum())
        shortage_total += float((shortage_cost * short).sum())
        units_short += float(short.sum())

    setups = float((orders > 0).sum())
    total_demand = float(demand.sum())

    return {
        "holding_cost": holding_total,
        "shortage_cost": shortage_total,
        "setup_cost": setups * setup_cost,
        "total_cost": holding_total + shortage_total + setups * setup_cost,
        "order_lines": setups,
        "units_short": units_short,
        "fill_rate": 1.0 - units_short / total_demand if total_demand else 1.0,
    }
