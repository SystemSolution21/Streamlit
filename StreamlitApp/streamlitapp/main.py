from typing import Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit.runtime.uploaded_file_manager import UploadedFile

st.title(body="Simple Data Dashboard")

uploaded_file: UploadedFile | None = st.file_uploader(
    label="Choose a CSV file", type="csv"
)

if uploaded_file is not None:
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=uploaded_file)

    st.subheader(body="Data Preview")
    index_range: pd.RangeIndex = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
    df.index = index_range
    st.write(df)

    st.subheader(body="Data Summary")
    st.write(df.describe())

    st.subheader(body="Filter Data")
    columns: list[str] = df.columns.tolist()
    selected_column: str | None = st.selectbox(
        label="Select column to filter by", options=columns
    )
    unique_values: np.ndarray[Any, Any] = df[selected_column].unique()
    selected_value = st.selectbox(label="Select value", options=unique_values)

    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", columns)
    y_column = st.selectbox("Select y-axis column", columns)

    if st.button("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_column)[y_column])
else:
    st.write("Waiting on file upload...")
