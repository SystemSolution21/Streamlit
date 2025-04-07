import pandas as pd


# Get S&P 500 symbols using yfinance
def get_sp500_symbols() -> dict[str, str]:
    """Get S&P 500 symbols using yfinance."""

    sp500: pd.DataFrame = pd.read_html(
        io="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return dict(zip(sp500["Security"], sp500["Symbol"]))
