from typing import Any, Union, Tuple, Literal
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile


# Set page config
st.set_page_config(page_title="Advanced Data Dashboard", page_icon="📊", layout="wide")

# Add custom CSS
st.markdown(
    body="""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title(body="📊 Advanced Data Dashboard")
st.markdown(body="Upload your CSV file to analyze and visualize your data.")

# File upload
uploaded_file: UploadedFile | None = st.file_uploader(
    label="Choose a CSV file", type="csv"
)

if uploaded_file is not None:
    # Load data
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=uploaded_file)

    # Sidebar controls
    st.sidebar.header(body="Dashboard Controls")

    # Date range filter if date column exists
    date_columns = df.select_dtypes(include=["datetime64"]).columns
    if len(date_columns) > 0 or "Date" in df.columns:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(arg=df["Date"])
            date_col: str | None = st.sidebar.selectbox(
                label="Select Date Column",
                options=date_columns if len(date_columns) > 0 else ["Date"],
            )
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = st.sidebar.date_input(
            label="Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            df = df[
                (df[date_col].dt.date >= date_range[0])
                & (df[date_col].dt.date <= date_range[1])
            ]

    # Main dashboard area using columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Data Overview")
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Quick Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select column for statistics", numeric_cols)

        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Mean", f"{df[selected_col].mean():.2f}")
        with metric_col2:
            st.metric("Median", f"{df[selected_col].median():.2f}")
        with metric_col3:
            st.metric("Std Dev", f"{df[selected_col].std():.2f}")
        with metric_col4:
            st.metric("Count", len(df))

    with col2:
        st.subheader("📉 Data Distribution")
        plot_type = st.selectbox(
            "Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot", "Line Plot"]
        )

        if plot_type == "Histogram":
            fig = px.histogram(df, x=selected_col)
        elif plot_type == "Box Plot":
            fig = px.box(df, y=selected_col)
        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("Select X axis", numeric_cols)
            y_col = st.selectbox("Select Y axis", numeric_cols)
            fig = px.scatter(df, x=x_col, y=y_col)
        else:  # Line Plot
            if "Date" in df.columns:
                fig = px.line(df, x="Date", y=selected_col)
            else:
                fig = px.line(df, y=selected_col)

        st.plotly_chart(fig, use_container_width=True)

    # Advanced Analysis Section
    st.subheader("🔍 Advanced Analysis")
    analysis_tabs = st.tabs(["Correlation Matrix", "Group Analysis", "Trend Analysis"])

    with analysis_tabs[0]:
        # Correlation Matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix, title="Correlation Matrix", color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)

    with analysis_tabs[1]:
        # Group Analysis
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            group_col = st.selectbox("Group by", categorical_cols)
            agg_col = st.selectbox("Aggregate column", numeric_cols)
            agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count"])

            grouped_data = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
            fig = px.bar(
                grouped_data,
                x=group_col,
                y=agg_col,
                title=f"{agg_func.capitalize()} of {agg_col} by {group_col}",
            )
            st.plotly_chart(fig, use_container_width=True)

    with analysis_tabs[2]:
        # Trend Analysis
        if "Date" in df.columns:
            trend_col = st.selectbox("Select column for trend analysis", numeric_cols)
            trend_period = st.selectbox(
                "Select trend period", ["Daily", "Weekly", "Monthly"]
            )

            if trend_period == "Daily":
                trend_data = df.groupby("Date")[trend_col].mean().reset_index()
            elif trend_period == "Weekly":
                trend_data = (
                    df.groupby(pd.Grouper(key="Date", freq="W"))[trend_col]
                    .mean()
                    .reset_index()
                )
            else:
                trend_data = (
                    df.groupby(pd.Grouper(key="Date", freq="M"))[trend_col]
                    .mean()
                    .reset_index()
                )

            fig = px.line(
                trend_data,
                x="Date",
                y=trend_col,
                title=f"{trend_period} Trend of {trend_col}",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.subheader("📥 Export Options")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="dashboard_export.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Export to Excel"):
            df.to_excel("dashboard_export.xlsx", index=False)
            with open("dashboard_export.xlsx", "rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f,
                    file_name="dashboard_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

else:
    st.info("Please upload a CSV file to begin analysis.")

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
