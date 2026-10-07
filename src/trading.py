"""GGR trading rule, pair returns and the six portfolio return series."""
import numpy as np
import pandas as pd

from config import BORROW_FEE_ANNUAL, COST_PER_LEG, N_PAIRS, TRADING_DAYS, TRIGGER_SD
from formation import PORTFOLIOS, periods


def trade_pair(px: pd.DataFrame, i: str, j: str, sigma: float, dates: pd.DatetimeIndex,
               wait: int) -> tuple[np.ndarray, list[dict]]:
    """Daily net return and trades of pair (i, j) over E_k, Eqs (5) and (6)."""
    P = px.loc[dates, [i, j]].to_numpy()
    d = P[:, 0] / P[0, 0] - P[:, 1] / P[0, 1]
    r = np.zeros(len(dates))
    trades, n_prev, o = [], 0, None
    for t in range(len(dates)):
        if n_prev != 0:
            r[t] = n_prev * ((P[t, 0] - P[t - 1, 0]) / P[o, 0] - (P[t, 1] - P[t - 1, 1]) / P[o, 1])
            short = 1 if n_prev > 0 else 0
            r[t] -= BORROW_FEE_ANNUAL / TRADING_DAYS * P[t - 1, short] / P[o, short]
        spread = d[t - wait] if t >= wait else 0.0
        if t == len(dates) - 1:
            n = 0
        elif n_prev == 0:
            n = -int(np.sign(spread)) if abs(spread) > TRIGGER_SD * sigma else 0
        else:
            n = 0 if spread * n_prev >= 0 else n_prev
        r[t] -= 2 * COST_PER_LEG * abs(n - n_prev)
        if n_prev == 0 and n != 0:
            o = t
        elif n_prev != 0 and n == 0:
            trades.append({"profit": r[o:t + 1].sum(), "converged": bool(spread * n_prev >= 0)})
        n_prev = n
    return r, trades


def backtest(px: pd.DataFrame, sel: pd.DataFrame,
             wait: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns of each (q, score) portfolio over all E_k, and every trade."""
    E = [e for _, e in periods(px.index)]
    returns = pd.DataFrame(0.0, index=E[0].append(E[1:]), columns=PORTFOLIOS)
    trades = []
    for pair in sel.itertuples():
        r, pair_trades = trade_pair(px, pair.i, pair.j, pair.sigma, E[pair.k], wait)
        returns.loc[E[pair.k], (pair.q, pair.score)] += r / N_PAIRS
        trades += [{"q": pair.q, "score": pair.score, **trade} for trade in pair_trades]
    return returns, pd.DataFrame(trades)
