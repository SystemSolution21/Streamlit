import re
from re import Match
from typing import Any, Sequence, Union, Tuple, cast, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.delta_generator import DeltaGenerator
from pandas.core.groupby import DataFrameGroupBy
import os
from pathlib import Path
import news_sentiment_openai
import news_sentiment

# Set page config
st.set_page_config(page_title="S&P 500 Stocks-Dashboard", page_icon="📊", layout="wide")

# Get the current file's directory
current_dir: Path = Path(__file__).parent


# Load CSS
def load_css(css_file) -> None:
    with open(file=current_dir / css_file) as f:
        st.markdown(body=f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Get S&P 500 symbols
def get_sp500_symbols() -> dict[str, str]:
    """Get S&P 500 symbols."""
    sp500: pd.DataFrame = pd.read_html(
        io="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return dict(zip(sp500["Security"], sp500["Symbol"]))


# Fetch stock data
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data for S&P 500 symbols."""
    ticker = ticker.lower() + ".us"
    end: datetime = datetime.today()
    start: datetime = end - timedelta(days=5 * 365)
    url: str = (
        f"https://stooq.com/q/d/l/?s={ticker}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    )

    df: pd.DataFrame = (
        pd.read_csv(filepath_or_buffer=url, parse_dates=["Date"])
        .set_index(keys="Date")
        .sort_index()
    )
    return df


# Load the CSS file
load_css(css_file="styles/dashboard.css")

# Title and description
st.title(body="📊 Stocks Visualize & Analyze News Sentiment")
st.sidebar.header(body="S&P 500 Stocks Dashboard")

# Data source selection
data_source: str = st.sidebar.radio(
    label="Upload a Stock CSV/Excel file or Fetch Stock Data to analyze and visualize.",
    options=["Upload Stock CSV/Excel", "Fetch Stock Data"],
)

# Initialize DataFrame
df: pd.DataFrame = pd.DataFrame()

if data_source == "Upload Stock CSV/Excel":

    # Upload stocks csv or excel file
    uploaded_file: UploadedFile | None = st.file_uploader(
        label="Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(filepath_or_buffer=uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(io=uploaded_file)
        except Exception as e:
            st.error(body=f"Error loading data: {e}")

        # Extract company name and symbol from the filename
        file_name_without_ext: str = os.path.splitext(uploaded_file.name)[0]

        company_name: str = ""
        symbol: str = ""

        last_paren_open_index: int = file_name_without_ext.rfind("(")
        last_paren_close_index: int = file_name_without_ext.rfind(")")

        if (
            0 <= last_paren_open_index < last_paren_close_index
            and last_paren_close_index == len(file_name_without_ext) - 1
        ):
            # Potential symbol is content within the last parentheses
            potential_symbol: str = file_name_without_ext[
                last_paren_open_index + 1 : last_paren_close_index
            ].strip()

            # Check if the potential symbol is all uppercase and not empty
            if potential_symbol and potential_symbol.isupper():
                symbol = potential_symbol
                company_name = file_name_without_ext[:last_paren_open_index].strip()
            else:
                # If not uppercase, treat the whole string as company name
                company_name = file_name_without_ext.strip()
        else:
            # Fallback if pattern (...) at the end is not found
            company_name = file_name_without_ext.strip()

        # Check company name and symbol
        if not company_name or not symbol:
            st.error(
                body=f"Invalid file name format. Should be in 'Company Name(SYMBOL).csv' format."
            )
            # Reset DataFrame to empty to prevent further processing
            df = pd.DataFrame()
            # Early return or continue to next iteration
            st.stop()

        # Validate that the file contains the expected columns for stock data
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(
                body=f"Invalid stock data format. Missing required columns: {', '.join(missing_columns)}. "
                f"Please ensure your file contains all of these columns: {', '.join(required_columns)}."
            )
            # Reset DataFrame to empty to prevent further processing
            df = pd.DataFrame()
            # Early return
            st.stop()

        # Create stock dictionary for company name and symbol
        stocks: dict[str, str] = {company_name: symbol}

        # Display selected company
        selected_company: str = st.sidebar.selectbox(
            label="Select a company",
            options=list(stocks.keys()),
            format_func=lambda x: (
                f"{x} ({stocks[x]})" if x in stocks and stocks[x] else x
            ),
        )

elif data_source == "Fetch Stock Data":

    # Get S&P 500 symbols and Fetch stock data
    try:
        # Get stock symbols and create dropdown
        stocks: dict[str, str] = get_sp500_symbols()
        selected_company: str = st.sidebar.selectbox(
            label="Select a company",
            options=list(stocks.keys()),
            format_func=lambda x: f"{x} ({stocks[x]})",
        )

        if selected_company:
            # Fetch stock data
            with st.spinner(text="Fetching stock data..."):
                df = fetch_stock_data(ticker=stocks[selected_company])
                df = (
                    df.reset_index()
                )  # Convert index to column (Index/Date/Close/Open/High/Low/Volume)

    except Exception as e:
        st.error(body=f"Error fetching stock data: {e}")

if not df.empty:
    # Date range filter if date column exists
    date_columns: pd.Index = df.select_dtypes(include=["datetime64"]).columns

    if len(date_columns) > 0 or "Date" in df.columns:
        # Process Date Column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(arg=df["Date"])  # 2025-05-10
            date_col: str = "Date"  # Use Date column directly without selection
        else:
            date_col: str = date_columns[
                0
            ]  # Use first datetime column if Date not present

        # Select Date Range
        min_date: datetime = df[date_col].min()
        max_date: datetime = df[date_col].max()

        date_range = st.sidebar.date_input(
            label="Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        # Filter data based on date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            df = df[
                (df[date_col].dt.date >= date_range[0])
                & (df[date_col].dt.date <= date_range[1])
            ]

    # Main dashboard area using columns
    col: List[DeltaGenerator] = st.columns(spec=2)
    with col[0]:
        st.subheader(body="📈 Data Overview")
        index_range: pd.RangeIndex = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        df.index = index_range
        st.dataframe(data=df, use_container_width=True)

        st.subheader(body="📊 Quick Statistics")
        numeric_cols: pd.Index = df.select_dtypes(include=[np.number]).columns
        selected_col: str | None = st.selectbox(
            label="Select column for statistics", options=numeric_cols
        )

        # Display metrics in columns
        metric_col: List[DeltaGenerator] = st.columns(spec=4)
        with metric_col[0]:
            st.metric(label="Mean", value=f"{df[selected_col].mean():.2f}")
        with metric_col[1]:
            st.metric(label="Median", value=f"{df[selected_col].median():.2f}")
        with metric_col[2]:
            st.metric(label="Std Dev", value=f"{df[selected_col].std():.2f}")
        with metric_col[3]:
            st.metric(label="Count", value=len(df))

    with col[1]:
        st.subheader(body="📉 Data Distribution")
        plot_type: str | None = st.selectbox(
            label="Select Plot Type",
            options=["Histogram", "Box Plot", "Scatter Plot", "Line Plot"],
        )

        match plot_type:
            case "Histogram":
                fig: go.Figure = px.histogram(data_frame=df, x=selected_col)
            case "Box Plot":
                fig = px.box(data_frame=df, y=selected_col)
            case "Scatter Plot":
                x_col: str | None = st.selectbox(
                    label="Select X axis", options=numeric_cols
                )
                y_col: str | None = st.selectbox(
                    label="Select Y axis", options=numeric_cols
                )
                fig = px.scatter(data_frame=df, x=x_col, y=y_col)
            case "Line Plot":
                if "Date" in df.columns:
                    fig = px.line(data_frame=df, x="Date", y=selected_col)
                else:
                    fig = px.line(data_frame=df, y=selected_col)
            case _:
                raise ValueError(f"Unsupported plot type: {plot_type}")

        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    # Advanced Analysis Section
    st.subheader(body="🔍 Advanced Analysis")
    analysis_tabs: Sequence[DeltaGenerator] = st.tabs(
        tabs=["Correlation Matrix", "Group Analysis", "Trend Analysis"]
    )

    with analysis_tabs[0]:
        # Correlation Matrix
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
        corr_matrix: pd.DataFrame = numeric_df.corr()
        fig = px.imshow(
            img=corr_matrix, title="Correlation Matrix", color_continuous_scale="RdBu"
        )
        st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with analysis_tabs[1]:
        # Group Analysis
        categorical_cols: pd.Index = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:

            # Select group and aggregation columns
            group_col: str | None = st.selectbox(
                label="Group by", options=categorical_cols
            )
            agg_col: str | None = st.selectbox(
                label="Aggregate column", options=numeric_cols
            )
            agg_func: str | None = cast(
                str,
                st.selectbox(
                    label="Aggregation function", options=["mean", "sum", "count"]
                ),
            )

            # Perform groupby and aggregation
            grouped_data: pd.DataFrame = (
                df.groupby(by=group_col)[[agg_col]].agg(func=agg_func).reset_index()
            )

            # Plot the grouped data
            fig = px.bar(
                data_frame=grouped_data,
                x=group_col,
                y=agg_col,
                title=f"{agg_func.capitalize()} of {agg_col} by {group_col}",
            )
            st.plotly_chart(figure_or_data=fig, use_container_width=True)

    with analysis_tabs[2]:
        # Trend Analysis
        if "Date" in df.columns:
            trend_col: Any | None = st.selectbox(
                label="Select column for trend analysis", options=numeric_cols
            )
            trend_period: str | None = st.selectbox(
                label="Select trend period", options=["Daily", "Weekly", "Monthly"]
            )

            if trend_period == "Daily":
                trend_data: pd.DataFrame = (
                    df.groupby(by="Date")[[trend_col]].mean().reset_index()
                )
            elif trend_period == "Weekly":
                trend_data = (
                    df.groupby(by=pd.Grouper(key="Date", freq="W"))[[trend_col]]
                    .mean()
                    .reset_index()
                )
            else:
                trend_data = (
                    df.groupby(by=pd.Grouper(key="Date", freq="M"))[[trend_col]]
                    .mean()
                    .reset_index()
                )

            fig = px.line(
                data_frame=trend_data,
                x="Date",
                y=trend_col,
                title=f"{trend_period} Trend of {trend_col}",
            )
            st.plotly_chart(figure_or_data=fig, use_container_width=True)

    # Export options
    st.subheader(body="📥 Export Options")
    exp_opt_col: List[DeltaGenerator] = st.columns(spec=2)
    with exp_opt_col[0]:
        if st.button(label="Export to CSV"):
            csv: str | None = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_company}({stocks[selected_company]}).csv",
                mime="text/csv",
            )

    with exp_opt_col[1]:
        if st.button(label="Export to Excel"):
            # Define the dynamic filename based on the selected company and its symbol
            excel_file_name: str = (
                f"{selected_company}({stocks[selected_company]}).xlsx"
            )

            # Save the DataFrame to an Excel file on the server with the dynamic name
            df.to_excel(excel_writer=excel_file_name, index=False)

            # Open the newly created server-side file in binary read mode
            with open(file=excel_file_name, mode="rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f,
                    file_name=excel_file_name,  # Suggest the same dynamic name for download
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            # Optional: Clean up the temporary file from the server after download is initiated
            try:
                os.remove(path=excel_file_name)
            except OSError as e:
                st.warning(
                    body=f"Could not remove temporary server file {excel_file_name}: {e}"
                )
else:
    st.info(body="Please upload a CSV/Excel file to begin analysis.")

    # Data preview
    st.subheader(body="📋 Data Format")
    st.markdown(
        body="""
    CSV/Excel file should have columns like:
    - Date (YYYY-MM-DD)
    - Numeric columns for analysis
    - Categorical columns for grouping
    
    Sample data available: 
    - Apple Inc.(AAPL).csv
    - Alphabet Inc. (Class A)(GOOGL).xlsx
    """
    )
# Horizontal line
st.sidebar.html(body="<hr>")
# OpenAI API key
with st.sidebar:
    openai_api_key = st.text_input(
        label="Better Analysis, OpenAI API Key", key="openai_api_key", type="password"
    )

# Select OpenAI model
with st.sidebar:
    openai_model: str = st.selectbox(
        label="Select OpenAI Model",
        options=[
            "o4-mini-2025-04-16",
            "gpt-4.1-nano-2025-04-14",
            "o3-mini-2025-01-31",
            "gpt-4o-mini-2024-07-18",
        ],
        index=0,
    )

# Stock news sentiment analysis
if st.sidebar.button(
    label=f"Stock News Sentiment Analyze AI",
    help="Default local install Ollama llm",
):
    try:
        if selected_company:

            with st.spinner(
                text=f"AI Analyzing news sentiment for {selected_company} ({stocks[selected_company]})..."
            ):
                try:
                    if openai_api_key and openai_model:

                        # OpenAI API key and model
                        news_sentiment_openai.analyze_stock_news(
                            openai_api_key=openai_api_key,
                            openai_model=openai_model,
                            symbol=stocks[selected_company],
                        )

                    else:
                        # Default locally installed Ollama gemma3:4b
                        news_sentiment.analyze_stock_news(
                            symbol=stocks[selected_company]
                        )

                except Exception as e:
                    error_message = str(object=e)
                    # Handle OpenAI API key errors
                    if (
                        "invalid_api_key" in error_message
                        or "Incorrect API key" in error_message
                    ):
                        message_match: Match[str] | None = re.search(
                            pattern=r"'message':\s*'([^']*)'", string=error_message
                        )
                        if message_match:
                            error_message: str = message_match.group(1)

                    st.error(body=error_message)

    except NameError:
        st.error(
            body="Please upload a S&P 500 stocks CSV/Excel file or Fetch S&P 500 stocks data to analyze news sentiment."
        )

# Create some space before the footer
st.markdown(body="<br><br>", unsafe_allow_html=True)

# Add a footer with CSS to ensure it stays at the bottom
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 10px;
    font-size: 0.8rem;
    border-top: 1px solid #dee2e6;
    z-index: 1000;
}
</style>
<div class="footer">
    <p>© 2025 SystemSolution21. All rights reserved. S&P 500 Stock Dashboard</p>
</div>
"""
st.markdown(body=footer_html, unsafe_allow_html=True)

# Add padding at the bottom to prevent content from being hidden behind the footer
st.markdown(body="<div style='padding-bottom: 70px;'></div>", unsafe_allow_html=True)
