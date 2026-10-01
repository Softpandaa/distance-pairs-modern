"""Every parameter of the study, declared once."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURE = ROOT / "latex" / "cumulative.png"

FIRST_TRADE = "2014-06-01"   # first day of the first trading period E_1
N_PERIODS = 20               # six-month trading periods, Jun 2014 to May 2024
FORMATION_MONTHS = 12        # length of the formation period Phi_k
TRADING_MONTHS = 6           # length of the trading period E_k

DISTANCES = ("L2", "L1", "Linf")   # Euclidean, Manhattan, Chebyshev
SCORES = ("distance", "composite")  # m = 0, m = 1
WEIGHTS = (0.6, 0.2, 0.2)           # composite weights on 1/D, NZC, sigma
N_PAIRS = 20                        # pairs per portfolio

DAYS = 252         # trading days per year, for annualization
TRIGGER = 2.0      # open when |d_t| exceeds TRIGGER formation standard deviations
COST = 0.001       # cost per dollar traded per leg, 10 basis points
BORROW = 0.0025    # short borrowing fee per year on the short leg value, D'Avolio (2002)
ROBUST_WAIT = 1    # robustness: trading days between signal and execution (baseline 0)
COVID = 11         # index k of the Dec 2019 to May 2020 trading period, reported separately
