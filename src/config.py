"""Every parameter of the study, declared once."""
from pathlib import Path

# Paths -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURE = ROOT / "latex" / "cumulative.png"

# Sample ----------------------------------------------------------------------
FIRST_TRADE = "2014-06-01"
N_PERIODS = 20
FORMATION_MONTHS = 12
TRADING_MONTHS = 6
COVID_PERIOD = 11                     # Dec 2019 to May 2020, counted from 0

# Pair formation --------------------------------------------------------------
DISTANCES = ("L2", "L1", "Linf")
SCORES = ("distance", "composite")
COMPOSITE_WEIGHTS = (0.6, 0.2, 0.2)   # on 1/D, NZC, sigma
N_PAIRS = 20

# Trading ---------------------------------------------------------------------
TRADING_DAYS = 252
TRIGGER_SD = 2.0
COST_PER_LEG = 0.001
BORROW_FEE_ANNUAL = 0.0025
WAIT_DAYS = 1                         # robustness run, baseline 0
