# Q4(a), Q5(a), and Q5(b): Streamlit Dashboard for PGCB Demand Forecasting and Monitoring

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PGCB Demand Forecasting Dashboard",
    layout="wide"
)

st.title("PGCB Electricity Demand Forecasting Dashboard")
st.write(
    "This dashboard supports stakeholder decision-making by showing demand trends, "
    "model predictions, forecast error patterns, monitoring metrics, and performance degradation analysis "
    "from the best model selected in Q2."
)


@st.cache_data
def load_data():
    data = pd.read_csv("dashboard_data.csv")
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["date"] = data["datetime"].dt.date
    return data


def calculate_metrics(data):
    mae = data["Absolute Error MW"].mean()

    rmse = np.sqrt(
        np.mean(
            (data["Actual Demand MW"] - data["Predicted Demand MW"]) ** 2
        )
    )

    mape = (
        abs(data["Actual Demand MW"] - data["Predicted Demand MW"])
        / data["Actual Demand MW"]
    ).mean() * 100

    return mae, rmse, mape


df = load_data()

# Sidebar interactive filters
st.sidebar.header("Dashboard Filters")

date_range = st.sidebar.slider(
    "Select date range",
    min_value=df["date"].min(),
    max_value=df["date"].max(),
    value=(df["date"].min(), df["date"].max())
)

selected_hours = st.sidebar.multiselect(
    "Select hour(s)",
    options=sorted(df["hour"].unique()),
    default=sorted(df["hour"].unique())
)

show_peak_only = st.sidebar.checkbox(
    "Show peak-hour records only",
    value=False
)

filtered_df = df[
    (df["date"] >= date_range[0]) &
    (df["date"] <= date_range[1]) &
    (df["hour"].isin(selected_hours))
].copy()

if show_peak_only:
    filtered_df = filtered_df[filtered_df["is_peak_hour"] == 1]

if filtered_df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

filtered_df = filtered_df.sort_values("datetime").reset_index(drop=True)

# Predictive and analytical output
latest_row = filtered_df.iloc[-1]

avg_error, overall_rmse, mape = calculate_metrics(filtered_df)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Predicted Demand",
    f"{latest_row['Predicted Demand MW']:,.0f} MW"
)

col2.metric(
    "Average Absolute Error",
    f"{avg_error:,.0f} MW"
)

col3.metric(
    "MAPE",
    f"{mape:.2f}%"
)

# Q5(a): Monitoring Metrics Section
st.divider()
st.subheader("Monitoring Metrics")

# Model performance monitoring
monitor_mae, monitor_rmse, monitor_mape = calculate_metrics(filtered_df)

# Data quality monitoring
missing_cells = int(filtered_df.isna().sum().sum())
duplicate_timestamps = int(filtered_df.duplicated(subset=["datetime"]).sum())

m1, m2, m3, m4 = st.columns(4)

m1.metric("Model MAE", f"{monitor_mae:,.0f} MW")
m2.metric("Model RMSE", f"{monitor_rmse:,.0f} MW")
m3.metric("Missing Cells", missing_cells)
m4.metric("Duplicate Timestamps", duplicate_timestamps)

monitoring_table = pd.DataFrame({
    "Monitoring Metric": [
        "Model MAE",
        "Model RMSE",
        "Missing Cell Count",
        "Duplicate Timestamp Count"
    ],
    "Metric Type": [
        "Model Performance",
        "Model Performance",
        "Data Quality",
        "Data Quality"
    ],
    "What Is Being Monitored": [
        "Average prediction error in megawatts.",
        "Larger prediction errors in the selected dashboard data.",
        "Number of missing cells in the filtered dashboard data.",
        "Number of repeated datetime values in the filtered dashboard data."
    ],
    "Current Result": [
        f"{monitor_mae:,.0f} MW",
        f"{monitor_rmse:,.0f} MW",
        missing_cells,
        duplicate_timestamps
    ]
})

st.dataframe(monitoring_table, use_container_width=True)

# Q5(b): Model Performance Degradation Analysis
st.divider()
st.subheader("Model Performance Degradation Analysis")

if len(filtered_df) < 20:
    st.warning(
        "Not enough records are available for performance degradation analysis. "
        "Please select a wider date range."
    )
else:
    split_point = len(filtered_df) // 2

    reference_df = filtered_df.iloc[:split_point].copy()
    current_df = filtered_df.iloc[split_point:].copy()

    ref_mae, ref_rmse, ref_mape = calculate_metrics(reference_df)
    cur_mae, cur_rmse, cur_mape = calculate_metrics(current_df)

    mae_change = ((cur_mae - ref_mae) / ref_mae) * 100 if ref_mae != 0 else 0
    rmse_change = ((cur_rmse - ref_rmse) / ref_rmse) * 100 if ref_rmse != 0 else 0
    mape_change = ((cur_mape - ref_mape) / ref_mape) * 100 if ref_mape != 0 else 0

    degradation_status = (
        "Degradation Detected"
        if (mae_change > 10 or rmse_change > 10 or mape_change > 10)
        else "Stable"
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Current MAE",
        f"{cur_mae:,.0f} MW",
        delta=f"{mae_change:.2f}% vs reference"
    )

    d2.metric(
        "Current RMSE",
        f"{cur_rmse:,.0f} MW",
        delta=f"{rmse_change:.2f}% vs reference"
    )

    d3.metric(
        "Current MAPE",
        f"{cur_mape:.2f}%",
        delta=f"{mape_change:.2f}% vs reference"
    )

    d4.metric(
        "Performance Status",
        degradation_status
    )

    degradation_table = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE"],
        "Reference Period": [
            f"{ref_mae:,.0f} MW",
            f"{ref_rmse:,.0f} MW",
            f"{ref_mape:.2f}%"
        ],
        "Current Period": [
            f"{cur_mae:,.0f} MW",
            f"{cur_rmse:,.0f} MW",
            f"{cur_mape:.2f}%"
        ],
        "Percentage Change": [
            f"{mae_change:.2f}%",
            f"{rmse_change:.2f}%",
            f"{mape_change:.2f}%"
        ],
        "Interpretation": [
            "Higher MAE means average forecast error increased.",
            "Higher RMSE means larger forecast errors became more serious.",
            "Higher MAPE means percentage forecast error increased."
        ]
    })

    st.dataframe(degradation_table, use_container_width=True)

    chart_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE"],
        "Reference Period": [ref_mae, ref_rmse, ref_mape],
        "Current Period": [cur_mae, cur_rmse, cur_mape]
    }).set_index("Metric")

    st.bar_chart(chart_df)

    st.write(
        "This analysis compares the earlier part of the selected dashboard data with the most recent part. "
        "If the current period has higher errors than the reference period, the model may be showing performance degradation. "
        "This can affect future model performance because the model may become less reliable when demand patterns change."
    )

st.divider()

# Visualization 1
st.subheader("Visualization 1: Actual vs Predicted Demand")

line_data = filtered_df[
    ["datetime", "Actual Demand MW", "Predicted Demand MW"]
].set_index("datetime")

st.line_chart(line_data)

# Visualization 2
st.subheader("Visualization 2: Average Demand by Hour")

hourly_avg = (
    filtered_df.groupby("hour")["Actual Demand MW"]
    .mean()
    .reset_index()
    .rename(columns={"Actual Demand MW": "Average Demand MW"})
)

st.bar_chart(hourly_avg.set_index("hour"))

# Visualization 3
st.subheader("Visualization 3: Prediction Error Distribution")

if filtered_df["Absolute Error MW"].nunique() > 1:
    counts, edges = np.histogram(filtered_df["Absolute Error MW"], bins=15)

    hist_df = pd.DataFrame({
        "Error Range": [
            f"{int(edges[i])}-{int(edges[i + 1])} MW"
            for i in range(len(edges) - 1)
        ],
        "Number of Records": counts
    })

    st.bar_chart(hist_df.set_index("Error Range"))
else:
    st.info("Not enough error variation to show a distribution.")

st.divider()

st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)
