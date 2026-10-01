"""Load the committed prices and sectors."""
import pandas as pd

from config import DATA


def prices() -> pd.DataFrame:
    """Adjusted closes, one column per ticker."""
    return pd.read_csv(DATA / "prices.csv.gz", index_col=0, parse_dates=True)


def sectors() -> pd.Series:
    """Sector of each ticker."""
    return pd.read_csv(DATA / "sectors.csv", index_col=0)["sector"]
