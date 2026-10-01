"""Single entry point: prints every table of the report and writes its figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import data
import stats
from config import COVID, DISTANCES, FIGURE, ROBUST_WAIT, SCORES
from formation import formation, periods
from trading import backtest

NAMES = {"L2": "Euclidean", "L1": "Manhattan", "Linf": "Chebyshev"}
STYLE = {"L2": ("#0072B2", "-"), "L1": ("#D55E00", "--"), "Linf": ("#009E73", ":")}  # Okabe-Ito


def figure(returns, covid):
    """Cumulative net return of the six portfolios, COVID trading period shaded."""
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, score in zip(axes, SCORES):
        ax.axvspan(covid[0], covid[-1], color="0.85")
        for q in DISTANCES:
            colour, dash = STYLE[q]
            ax.plot(100 * returns[(q, score)].cumsum(), color=colour, linestyle=dash,
                    linewidth=1.2, label=NAMES[q])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(score.capitalize())
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("Cumulative net return (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=300)


if __name__ == "__main__":
    px, sectors = data.prices(), data.sectors()
    sel = formation(px, sectors)
    returns, trades = backtest(px, sel)
    delayed, _ = backtest(px, sel, wait=ROBUST_WAIT)
    covid = periods(px.index)[COVID][1]
    pre, post = returns.index < covid[0], returns.index > covid[-1]
    n = sectors.value_counts()
    print(f"Sectors\n{n.to_frame('stocks').assign(pairs=n * (n - 1) // 2).to_string()}")
    for title, table in [
            ("Performance, H1: E[r] > 0 (one-sided p)", stats.performance(returns)),
            ("H2: composite minus distance (two-sided p)", stats.h2(returns)),
            ("H3: q minus L2 (two-sided p)", stats.h3(returns)),
            ("Mechanism", stats.mechanism(sel, trades)),
            (f"Before {covid[0]:%Y-%m-%d}", stats.performance(returns[pre])),
            (f"{covid[0]:%Y-%m-%d} to {covid[-1]:%Y-%m-%d}",
             stats.performance(returns[~pre & ~post])),
            (f"After {covid[-1]:%Y-%m-%d}", stats.performance(returns[post])),
            ("Excluding the COVID period", stats.performance(returns[pre | post])),
            (f"Robustness, execution {ROBUST_WAIT} day after the signal",
             stats.performance(delayed)),
            ("Robustness, H2", stats.h2(delayed))]:
        print(f"\n{title}\n{table.to_string(float_format='{:.3g}'.format)}")
    figure(returns, covid)
