import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from pandas_datareader import data as pdr
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set page config
st.set_page_config(
    page_title="Tokyo Stock Exchange Stock Analysis", page_icon="📈", layout="wide"
)

# Application title
st.title("Tokyo Stock Exchange Stock Analysis")

# Initialize session state for storing data
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

# Sidebar for user inputs
st.sidebar.header("Settings")

# Tokyo Stock Exchange tickers (Top 50 TSE stocks)
tse_tickers = {
    "Toyota Motor": "7203.T",
    "Sony Group": "6758.T",
    "Keyence": "6861.T",
    "Mitsubishi UFJ Financial": "8306.T",
    "Nintendo": "7974.T",
    "SoftBank Group": "9984.T",
    "Recruit Holdings": "6098.T",
    "Tokio Marine Holdings": "8766.T",
    "Shin-Etsu Chemical": "4063.T",
    "Hitachi": "6501.T",
    "KDDI": "9433.T",
    "Daiichi Sankyo": "4568.T",
    "Fast Retailing": "9983.T",
    "Mitsubishi Corporation": "8058.T",
    "Nippon Telegraph & Telephone": "9432.T",
    "Sumitomo Mitsui Financial": "8316.T",
    "Takeda Pharmaceutical": "4502.T",
    "Hoya": "7741.T",
    "Daikin Industries": "6367.T",
    "Mizuho Financial Group": "8411.T",
    "Tokyo Electron": "8035.T",
    "Nidec": "6594.T",
    "Canon": "7751.T",
    "Astellas Pharma": "4503.T",
    "Mitsui & Co": "8031.T",
    "Honda Motor": "7267.T",
    "Orix": "8591.T",
    "Japan Tobacco": "2914.T",
    "Murata Manufacturing": "6981.T",
    "Denso": "6902.T",
}

# Ticker selection
selected_company = st.sidebar.selectbox("Select a company:", list(tse_tickers.keys()))
selected_ticker = tse_tickers[selected_company]

# Date range selection
st.sidebar.subheader("Date Range")
today = datetime.date.today()
start_date = st.sidebar.date_input("Start date", today - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End date", today)

# Technical indicators selection
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=True)
sma_period = (
    st.sidebar.slider("SMA Period", min_value=5, max_value=200, value=20, step=5)
    if show_sma
    else 20
)

show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)")
ema_period = (
    st.sidebar.slider("EMA Period", min_value=5, max_value=200, value=20, step=5)
    if show_ema
    else 20
)

show_bollinger = st.sidebar.checkbox("Bollinger Bands")
bollinger_period = (
    st.sidebar.slider("Bollinger Period", min_value=10, max_value=50, value=20, step=2)
    if show_bollinger
    else 20
)
bollinger_std = (
    st.sidebar.slider("Standard Deviation", min_value=1, max_value=4, value=2, step=1)
    if show_bollinger
    else 2
)

show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)")
rsi_period = (
    st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14, step=1)
    if show_rsi
    else 14
)

show_macd = st.sidebar.checkbox("MACD")


# Function to fetch stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    try:
        yf.pdr_override()
        data = pdr.get_data_yahoo(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# Function to calculate technical indicators
def calculate_indicators(data):
    # Simple Moving Average
    if show_sma:
        data[f"SMA_{sma_period}"] = data["Close"].rolling(window=sma_period).mean()

    # Exponential Moving Average
    if show_ema:
        data[f"EMA_{ema_period}"] = (
            data["Close"].ewm(span=ema_period, adjust=False).mean()
        )

    # Bollinger Bands
    if show_bollinger:
        data[f"SMA_{bollinger_period}"] = (
            data["Close"].rolling(window=bollinger_period).mean()
        )
        data["STD"] = data["Close"].rolling(window=bollinger_period).std()
        data["Upper_Band"] = data[f"SMA_{bollinger_period}"] + (
            data["STD"] * bollinger_std
        )
        data["Lower_Band"] = data[f"SMA_{bollinger_period}"] - (
            data["STD"] * bollinger_std
        )

    # RSI
    if show_rsi:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    if show_macd:
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Histogram"] = data["MACD"] - data["Signal_Line"]

    return data


# Main function to create the Plotly candlestick chart
def create_candlestick_chart(data):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick",
        )
    )

    # Add SMA
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f"SMA_{sma_period}"],
                mode="lines",
                name=f"SMA ({sma_period})",
                line=dict(color="blue"),
            )
        )

    # Add EMA
    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f"EMA_{ema_period}"],
                mode="lines",
                name=f"EMA ({ema_period})",
                line=dict(color="orange"),
            )
        )

    # Add Bollinger Bands
    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Upper_Band"],
                mode="lines",
                name="Upper Bollinger Band",
                line=dict(color="green", dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Lower_Band"],
                mode="lines",
                name="Lower Bollinger Band",
                line=dict(color="red", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(0, 100, 80, 0.1)",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_company} ({selected_ticker}) Stock Price",
        yaxis_title="Price (JPY)",
        xaxis_title="Date",
        template="plotly_white",
        height=600,
        hovermode="x unified",
    )

    return fig


# Create additional plots for technical indicators
def create_technical_plots(data):
    figs = []

    # RSI Plot
    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
            )
        )

        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_width=2, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_width=2, line_dash="dash", line_color="green")

        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis_title="RSI Value",
            height=250,
            template="plotly_white",
        )
        figs.append(fig_rsi)

    # MACD Plot
    if show_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue"),
            )
        )

        fig_macd.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Signal_Line"],
                mode="lines",
                name="Signal Line",
                line=dict(color="red"),
            )
        )

        fig_macd.add_trace(
            go.Bar(
                x=data.index,
                y=data["MACD_Histogram"],
                name="Histogram",
                marker_color=np.where(data["MACD_Histogram"] >= 0, "green", "red"),
            )
        )

        fig_macd.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            yaxis_title="MACD Value",
            height=250,
            template="plotly_white",
        )
        figs.append(fig_macd)

    return figs


# Initialize Llama model for stock analysis
def initialize_llm():
    try:
        # Setup Llama model (adjust path and parameters based on your setup)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Note: You need to have Llama model installed locally
        # This path should be adjusted to where your model is located
        llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.gguf",  # Adjust to your model path
            temperature=0.1,
            max_tokens=2000,
            n_ctx=4096,
            top_p=0.95,
            callback_manager=callback_manager,
            verbose=False,
        )
        return llm
    except Exception as e:
        st.warning(f"Could not initialize LLM model: {e}")
        st.warning(
            "Analysis features will be disabled. Please install the Llama model locally."
        )
        return None


# Function to analyze stock data with Llama
def analyze_stock_data(data, company_name, ticker):
    llm = initialize_llm()

    if llm is None:
        return "LLM model not available. Please install the Llama model to enable analysis."

    # Prepare data for analysis
    current_price = data["Close"].iloc[-1]
    previous_price = data["Close"].iloc[-2]
    price_change = ((current_price - previous_price) / previous_price) * 100

    # Calculate 52-week high/low
    high_52week = data["High"].rolling(window=252).max().iloc[-1]
    low_52week = data["Low"].rolling(window=252).min().iloc[-1]

    # Calculate average volume
    avg_volume = data["Volume"].mean()

    # Calculate volatility (standard deviation of returns)
    returns = data["Close"].pct_change()
    volatility = returns.std() * 100

    # Prepare prompt for LLM
    prompt = f"""
    Analyze the following stock market data for {company_name} ({ticker}):
    
    Current Price: {current_price:.2f} JPY
    Daily Change: {price_change:.2f}%
    52-Week High: {high_52week:.2f} JPY
    52-Week Low: {low_52week:.2f} JPY
    Average Volume: {avg_volume:.0f}
    Volatility: {volatility:.2f}%
    
    Provide a concise analysis of the stock's performance, potential outlook, and key factors to consider.
    Your analysis should be objective and focus on the technical aspects visible from the data.
    Limit your response to 3-4 paragraphs.
    """

    # Get LLM response
    response = llm(prompt)
    return response


# Main application flow
try:
    # Load data
    data = get_stock_data(selected_ticker, start_date, end_date)

    if data is not None and not data.empty:
        st.session_state.stock_data = data

        # Calculate indicators
        data_with_indicators = calculate_indicators(data.copy())

        # Display stock information
        col1, col2, col3, col4 = st.columns(4)

        current_price = data["Close"].iloc[-1]
        previous_price = data["Close"].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100

        col1.metric("Current Price", f"{current_price:.2f} JPY", f"{price_change:.2f}%")
        col2.metric("Opening Price", f"{data['Open'].iloc[-1]:.2f} JPY")
        col3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        col4.metric(
            "52-Week Range", f"{data['Low'].min():.2f} - {data['High'].max():.2f} JPY"
        )

        # Create main candlestick chart
        main_chart = create_candlestick_chart(data_with_indicators)
        st.plotly_chart(main_chart, use_container_width=True)

        # Display technical indicator plots if any
        tech_figs = create_technical_plots(data_with_indicators)
        for fig in tech_figs:
            st.plotly_chart(fig, use_container_width=True)

        # Stock Analysis with LLM section
        st.header("Stock Analysis")

        with st.expander("View Analysis"):
            with st.spinner("Analyzing stock data with Llama..."):
                try:
                    analysis = analyze_stock_data(
                        data, selected_company, selected_ticker
                    )
                    st.write(analysis)
                except Exception as e:
                    st.error(f"Error in LLM analysis: {e}")
                    st.info(
                        "Note: To use the LLM analysis feature, you need to have the Llama model installed locally."
                    )

    else:
        st.warning(
            "No data available for the selected date range. Please try different dates or another ticker."
        )

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check your inputs and try again.")

# Add footer
st.markdown("---")
st.caption(
    "Data source: Yahoo Finance. This app is for educational purposes only and not financial advice."
)
