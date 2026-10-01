"""Newey-West tests of H1-H3 and the mechanism table."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from config import DAYS, DISTANCES, SCORES


def nw_t(x: pd.Series) -> float:
    """Newey-West t of the mean of x: Bartlett kernel, lag floor(4 (T/100)^(2/9)),
    no small-sample correction."""
    lags = int(4 * (len(x) / 100) ** (2 / 9))
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.tvalues[0]


def _differences(d: pd.DataFrame) -> pd.DataFrame:
    """Annualized mean (%), NW t and two-sided p of each column of d."""
    rows = []
    for c in d:
        t = nw_t(d[c])
        rows.append({"diff %": 100 * DAYS * d[c].mean(), "t": t, "p": 2 * norm.sf(abs(t))})
    return pd.DataFrame(rows, index=d.columns)


def performance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized mean and volatility (%), Sharpe, NW t and one-sided H1 p of each portfolio."""
    rows = []
    for c in returns:
        r = returns[c]
        t = nw_t(r)
        rows.append({"mean %": 100 * DAYS * r.mean(), "vol %": 100 * np.sqrt(DAYS) * r.std(),
                     "Sharpe": np.sqrt(DAYS) * r.mean() / r.std(), "t": t, "p": norm.sf(t)})
    return pd.DataFrame(rows, index=returns.columns)


def h2(returns: pd.DataFrame) -> pd.DataFrame:
    """H2: composite minus distance, for each q."""
    return _differences(returns.xs("composite", axis=1, level="score")
                        - returns.xs("distance", axis=1, level="score"))


def h3(returns: pd.DataFrame) -> pd.DataFrame:
    """H3: L1 and Linf minus L2, for each score."""
    return _differences(returns[["L1", "Linf"]].sub(returns["L2"], level="score"))


def mechanism(sel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Mean formation sigma and NZC, trades per selected pair, share of trades converged (%)
    and mean net profit per trade (%) overall, if converged and if forced, per portfolio."""
    s, tr = sel.groupby(["q", "score"]), trades.groupby(["q", "score"])
    by = trades.groupby(["q", "score", "converged"])["profit"].mean().unstack()
    idx = pd.MultiIndex.from_product([DISTANCES, SCORES], names=["q", "score"])
    return pd.DataFrame({"sigma": s["sigma"].mean(), "NZC": s["NZC"].mean(),
                         "trades/pair": tr.size() / s.size(),
                         "converged %": 100 * tr["converged"].mean(),
                         "profit %": 100 * tr["profit"].mean(),
                         "converged profit %": 100 * by[True],
                         "forced profit %": 100 * by[False]}, index=idx)
