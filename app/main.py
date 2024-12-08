import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from windrose import WindroseAxes

# Import EDA functions from the provided eda.py logic
from eda import (
    load_data,
    clean_data,
    generate_summary_statistics,
    plot_correlation_heatmap,
    plot_time_series,
    plot_time_series_with_precipitation,
    plot_wind_rose,
    evaluate_cleaning_impact,
    plot_bubble_chart,
)

# Streamlit UI for Dashboard
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("Exploratory Data Analysis Dashboard")
st.sidebar.header("Select Dataset and Visualization Options")

# Predefined datasets for Benin, Togo, and Sierra Leone
data_options = {
  'Benin': 'app/datasets/benin-malanville.csv',
  'Sierra Leone': 'app/datasets/sierraleone_data.csv',
  'Togo': 'app/datasets/togo-dapaong_qc.csv'
}

# Dropdown to select the predefined data or upload new dataset
dataset_choice = st.sidebar.radio(
    "Choose Dataset",
    ["Predefined Datasets", "Upload Your Own"]
)

# Predefined datasets section
if dataset_choice == "Predefined Datasets":
    country = st.sidebar.selectbox("Choose a Country Dataset", list(data_options.keys()))
    if country:
        # Load the selected dataset
        with st.spinner(f"Loading data for {country}..."):
            raw_data = load_data(data_options[country])

# Upload own dataset section
elif dataset_choice == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Loading your uploaded data..."):
            # Read the uploaded CSV file directly
            raw_data = load_data(uploaded_file)

# Check if data is loaded successfully
if 'raw_data' in locals() and not raw_data.empty:
    # Display raw data if the user wants
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader(f"Raw Data ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        st.write(raw_data.head())

    # Clean data
    with st.spinner("Cleaning data..."):
        cleaned_data = clean_data(raw_data)

    if st.sidebar.checkbox("Show Cleaned Data"):
        st.subheader(f"Cleaned Data ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        st.write(cleaned_data.head())

    # Generate summary statistics
    if st.sidebar.checkbox("Summary Statistics"):
        st.subheader(f"Summary Statistics ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        summary_stats = generate_summary_statistics(cleaned_data)
        st.dataframe(summary_stats)

    # Plot correlation heatmap
    if st.sidebar.checkbox("Correlation Heatmap"):
        st.subheader(f"Correlation Heatmap ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        plot_correlation_heatmap(cleaned_data)
        st.pyplot()

    # Time series plot
    if st.sidebar.checkbox("Time Series"):
        column = st.sidebar.selectbox(
            "Select Column for Time Series",
            options=cleaned_data.select_dtypes(include=[np.number]).columns,
        )
        if column:
            st.subheader(f"Time Series: {column} ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
            plot_time_series(cleaned_data, column)
            st.pyplot()

    # Time series with precipitation
    if st.sidebar.checkbox("Time Series with Precipitation"):
        st.subheader(f"Time Series of GHI and Precipitation ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        plot_time_series_with_precipitation(cleaned_data)
        st.pyplot()

    # Wind rose plot
    if st.sidebar.checkbox("Wind Rose"):
        st.subheader(f"Wind Rose ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        plot_wind_rose(cleaned_data)
        st.pyplot()

    # Evaluate cleaning impact
    if st.sidebar.checkbox("Cleaning Impact Evaluation"):
        st.subheader(f"Cleaning Impact ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        evaluate_cleaning_impact(cleaned_data)
        st.pyplot()

    # Bubble chart
    if st.sidebar.checkbox("Bubble Chart"):
        st.subheader(f"Bubble Chart: GHI vs Tamb vs WS ({'Uploaded' if dataset_choice == 'Upload Your Own' else country})")
        plot_bubble_chart(cleaned_data)
        st.pyplot()

else:
    st.info("Please select or upload a dataset to proceed.")
