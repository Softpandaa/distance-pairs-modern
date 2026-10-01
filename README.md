# Does the Distance Approach to Pairs Trading Survive in the Modern Era?

This study tests whether the distance approach to pairs trading of Gatev, Goetzmann and Rouwenhorst (2006) remains profitable for US large-cap stocks over the last decade, from June 2014 to May 2024. Pairs are formed within sectors under the Euclidean, Manhattan and Chebyshev distances, ranked either by the distance alone or by a composite score that also rewards zero crossings and spread volatility, and traded over 20 six-month periods net of transaction and borrowing costs.

## Findings

The distance approach does not survive in the modern era. No portfolio earns a significant return, with annualized means between -0.79% and 0.44%, and neither the composite score nor the choice of distance makes a significant difference. The COVID half-year from December 2019 to May 2020 is the best period for every portfolio, and outside it all six portfolios lose money.

## Layout

```
data/     committed input prices and sectors
src/      analysis modules, every parameter declared once in config.py
report.pdf
```

## Data

`data/prices.csv.gz` holds the daily adjusted closing prices of 189 US large-cap stocks from Yahoo Finance. `data/sectors.csv` holds the sector of each stock.

## Reproducing

Python 3.13.

```
pip install -r requirements.txt
python src/main.py
```

This prints every table of the report and writes its figure to `latex/cumulative.png`. The `latex/` folder is created on the first run and is not tracked.
