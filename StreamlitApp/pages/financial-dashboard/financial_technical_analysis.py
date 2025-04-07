from typing import Any, List, Tuple
import streamlit as st
import pandas as pd
from stocks import get_sp500_symbols
from datetime import datetime, timedelta, date
import yfinance as yf
import plotly.graph_objects as pgo
import tempfile
import base64
import os


# Set page config
st.set_page_config(
    page_title="Financial Technical Analysis", page_icon="📈", layout="wide"
)
st.title(body="Financial Technical Analysis")
st.sidebar.header(body="Stock Selection")


# Get stock symbols and create dropdown
stocks: dict[Any, Any] = get_sp500_symbols()
selected_company: Any | None = st.sidebar.selectbox(
    label="Select a company",
    options=list(stocks.keys()),
    format_func=lambda x: f"{x} ({stocks[x]})",
)

# Stock ticker and date range selection
ticker: str = stocks[selected_company]
start_date: date = st.sidebar.date_input(
    label="Start Date", value=datetime.now() - timedelta(days=365)
)
end_date: date = st.sidebar.date_input(label="End Date", value=datetime.now())


# Fetch stock data using yfinance
if st.sidebar.button(label="Fetch Data"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, ignore_tz=True)
        if data.empty:
            st.error(
                f"No data available for {selected_company} ({ticker}). The stock might be delisted or there could be an issue with the data provider."
            )
        else:
            st.session_state["stock_data"] = data
            st.success(body=f"Data for {selected_company} fetched successfully!")
    except Exception as e:
        st.error(f"Error fetching data for {selected_company} ({ticker}): {str(e)}")
        st.info("Please try another company or time period.")


# Display fetched data
if "stock_data" in st.session_state:
    st.subheader(body=selected_company)
    data = st.session_state["stock_data"]
    st.dataframe(data=data)
