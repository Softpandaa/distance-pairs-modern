"""Periods, eligible pairs, formation statistics and the six selected sets."""
from itertools import combinations

import numpy as np
import pandas as pd

from config import (DISTANCES, FIRST_TRADE, FORMATION_MONTHS, N_PAIRS,
                    N_PERIODS, SCORES, TRADING_MONTHS, WEIGHTS)


def periods(dates: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Formation dates Phi_k and trading dates E_k of each period k."""
    out = []
    for k in range(N_PERIODS):
        e0 = pd.Timestamp(FIRST_TRADE) + pd.DateOffset(months=TRADING_MONTHS * k)
        e1 = e0 + pd.DateOffset(months=TRADING_MONTHS)
        f0 = e0 - pd.DateOffset(months=FORMATION_MONTHS)
        out.append((dates[(dates >= f0) & (dates < e0)], dates[(dates >= e0) & (dates < e1)]))
    return out


def statistics(px: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """sigma, NZC and the three distances of every eligible within-sector pair over Phi_k.

    A stock is eligible when its price record over Phi_k is complete.
    """
    px = px.dropna(axis=1)
    p = px / px.iloc[0]
    rows = []
    for _, names in sectors[px.columns].groupby(sectors[px.columns]):
        for i, j in combinations(sorted(names.index), 2):
            d = (p[i] - p[j]).to_numpy()
            s = np.sign(d[d != 0])
            rows.append({"i": i, "j": j,
                         "sigma": d.std(ddof=1),
                         "NZC": int((s[1:] != s[:-1]).sum()),
                         "L2": np.sqrt((d ** 2).sum()),
                         "L1": np.abs(d).sum(),
                         "Linf": np.abs(d).max()})
    return pd.DataFrame(rows)


def _minmax(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min())


def select(pair_stats: pd.DataFrame, q: str, score: str) -> pd.DataFrame:
    """Top N_PAIRS pairs by lowest D^(q) (distance) or highest R^(q) (composite)."""
    if score == "distance":
        key = -pair_stats[q]
    else:
        key = (WEIGHTS[0] * _minmax(1 / pair_stats[q]) + WEIGHTS[1] * _minmax(pair_stats["NZC"])
               + WEIGHTS[2] * _minmax(pair_stats["sigma"]))
    return pair_stats.loc[key.nlargest(N_PAIRS).index]


def formation(px: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """The selected pairs of all six portfolios in every period."""
    chosen = []
    for k, (phi, _) in enumerate(periods(px.index)):
        pair_stats = statistics(px.loc[phi], sectors)
        for q in DISTANCES:
            for score in SCORES:
                chosen.append(select(pair_stats, q, score).assign(k=k, q=q, score=score))
    return pd.concat(chosen, ignore_index=True)
