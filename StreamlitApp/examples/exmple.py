import yfinance as yf

dat = yf.Ticker("MSFT")
try:
    print(dat.info)
except Exception as e:
    print(f" Error: {e}")
