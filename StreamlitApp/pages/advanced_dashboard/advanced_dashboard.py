from typing import Any, Sequence, Union, Tuple, cast, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.delta_generator import DeltaGenerator
from pandas.core.groupby import DataFrameGroupBy
import os
from pathlib import Path

# Set page config
st.set_page_config(page_title="Advanced Data Dashboard", page_icon="📊", layout="wide")

# Get the current file's directory
current_dir: Path = Path(__file__).parent


# Load CSS
def load_css(css_file) -> None:
    with open(file=current_dir / css_file) as f:
        st.markdown(body=f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the CSS file
load_css(css_file="styles/dashboard.css")

# Title and description
st.title(body="📊 Advanced Data Dashboard")
st.markdown(body="Upload your CSV file to analyze and visualize your data.")

# Upload CSV data file
uploaded_file: UploadedFile | None = st.file_uploader(
    label="Choose a CSV file", type="csv"
)

if uploaded_file is not None:
    # Load data
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=uploaded_file)

    # Sidebar controls
    st.sidebar.header(body="Dashboard Controls")

    # Date range filter if date column exists
    date_columns: pd.Index = df.select_dtypes(include=["datetime64"]).columns

    if len(date_columns) > 0 or "Date" in df.columns:
        # Select Date Columns
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(arg=df["Date"])  # 2023-01-01
            date_col: str | None = st.sidebar.selectbox(
                label="Select Date Column",
                options=date_columns if len(date_columns) > 0 else ["Date"],
            )
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
                file_name="dashboard_export.csv",
                mime="text/csv",
            )

    with exp_opt_col[1]:
        if st.button(label="Export to Excel"):
            df.to_excel(excel_writer="dashboard_export.xlsx", index=False)
            with open(file="dashboard_export.xlsx", mode="rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f,
                    file_name="dashboard_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

else:
    st.info(body="Please upload a CSV file to begin analysis.")

    # Sample data preview
    st.subheader(body="📋 Sample Data Format")
    st.markdown(
        body="""
    Your CSV file should have columns like:
    - Date (YYYY-MM-DD format)
    - Numeric columns for analysis
    - Categorical columns for grouping
    
    Example data is available in `Sample_Data_for_Plotting_and_Filtering.csv`
    """
    )
