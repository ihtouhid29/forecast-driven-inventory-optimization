"""Central configuration.

Every tunable lives here so experiments are reproducible and no magic numbers
are buried in the modelling code.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- scope
# The full M5 dataset is 30,490 series. We scope to a single store so the
# whole pipeline runs on a laptop in minutes rather than hours. The method
# generalises unchanged; only the row count differs.
STORE_ID = "CA_1"

# Day 1 of the M5 calendar.
FIRST_DATE = "2011-01-29"

# ---------------------------------------------------------------- horizon
# M5 forecasts 28 days ahead. We hold out the final 28 days as the test set
# and the 28 days before that as the validation set.
HORIZON = 28
N_TRAIN_DAYS = 1941          # d_1 ... d_1941 in sales_train_evaluation.csv
TEST_START_DAY = N_TRAIN_DAYS - HORIZON + 1        # d_1914
VAL_START_DAY = TEST_START_DAY - HORIZON           # d_1886

# ---------------------------------------------------------------- inventory
# Cost structure for the newsvendor and (s,S) models.
#
# These are assumptions, not measurements: M5 gives shelf prices but no cost
# of goods, no warehousing rates and no idea what a stockout costs the
# retailer. They are set to plausible grocery-retail values and every headline
# result is re-run across a range of them in the sensitivity analysis, because
# a cost saving that only holds at one assumed critical ratio is not a result.
GROSS_MARGIN = 0.30              # unit cost = price * (1 - margin)
UNIT_HOLDING_COST_RATE = 0.25    # annual holding cost as a fraction of unit cost
STOCKOUT_PENALTY_MULT = 1.5      # multiple of lost margin charged per unit short
ORDER_LINE_COST = 1.0            # fixed cost of placing one SKU line on an order

# Shelf-space cost per unit per day, charged regardless of what the unit is
# worth. Without this the model is degenerate: both shortage cost and holding
# cost are proportional to price, price cancels in the critical ratio, and
# every SKU in the store gets the identical ratio 0.9895. A real warehouse
# does not work that way. A pallet of paper towels and a pallet of vitamins
# occupy the same rack; the cheap bulky one is far more expensive to hold per
# dollar of value, so it should be allowed to stock out more often. This term
# is what makes the critical ratio a per-SKU quantity rather than a constant.
SPACE_COST_PER_UNIT_DAY = 0.004
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7

# Protection interval: a periodic-review order must cover demand until the
# NEXT order arrives, which is the review period plus the lead time.
PROTECTION_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

# Quantiles we train the forecaster on. The newsvendor picks one of these
# by matching it to the critical ratio Cu / (Cu + Co).
QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.975, 0.99]

# ---------------------------------------------------------------- misc
SEED = 42
