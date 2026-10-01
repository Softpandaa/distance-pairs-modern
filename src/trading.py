"""GGR trading rule, pair returns and the six portfolio return series."""
import numpy as np
import pandas as pd

from config import BORROW, COST, DAYS, DISTANCES, N_PAIRS, SCORES, TRIGGER
from formation import periods


def trade_pair(px: pd.DataFrame, i: str, j: str, sigma: float, dates: pd.DatetimeIndex,
               wait: int) -> tuple[np.ndarray, list[dict]]:
    """Daily net return of pair (i, j) over trading dates E_k, and its trades.

    A signal from the close of t is executed at the close of t + wait.
    """
    P = px.loc[dates, [i, j]].to_numpy()
    d = P[:, 0] / P[0, 0] - P[:, 1] / P[0, 1]
    r = np.zeros(len(dates))
    trades, n, o = [], 0, None
    for t in range(len(dates)):
        if n != 0:
            r[t] = n * ((P[t, 0] - P[t - 1, 0]) / P[o, 0] - (P[t, 1] - P[t - 1, 1]) / P[o, 1])
            short = 1 if n > 0 else 0  # leg held short
            r[t] -= BORROW / DAYS * P[t - 1, short] / P[o, short]
        s = d[t - wait] if t >= wait else 0.0  # spread observed at the signal close
        if t == len(dates) - 1:
            m = 0
        elif n == 0:
            m = -int(np.sign(s)) if abs(s) > TRIGGER * sigma else 0
        else:
            m = 0 if s * n >= 0 else n
        r[t] -= 2 * COST * abs(m - n)
        if n == 0 and m != 0:
            o = t
        elif n != 0 and m == 0:
            trades.append({"profit": r[o:t + 1].sum(), "converged": bool(s * n >= 0)})
        n = m
    return r, trades


def backtest(px: pd.DataFrame, sel: pd.DataFrame,
             wait: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns of each (q, score) portfolio over all E_k, and every trade."""
    E = [e for _, e in periods(px.index)]
    cols = pd.MultiIndex.from_product([DISTANCES, SCORES], names=["q", "score"])
    returns = pd.DataFrame(0.0, index=E[0].append(E[1:]), columns=cols)
    trades = []
    for p in sel.itertuples():
        r, tr = trade_pair(px, p.i, p.j, p.sigma, E[p.k], wait)
        returns.loc[E[p.k], (p.q, p.score)] += r / N_PAIRS
        trades += [{"q": p.q, "score": p.score, **t} for t in tr]
    return returns, pd.DataFrame(trades)
