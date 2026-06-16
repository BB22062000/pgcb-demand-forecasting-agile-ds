# Q4(a)-2: Streamlit Dashboard for PGCB Demand Forecasting

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
    "model predictions, and forecast error patterns from the best model selected in Q2."
)

@st.cache_data
def load_data():
    data = pd.read_csv("dashboard_data.csv")
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["date"] = data["datetime"].dt.date
    return data

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
]

if show_peak_only:
    filtered_df = filtered_df[filtered_df["is_peak_hour"] == 1]

if filtered_df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

# Predictive and analytical output
latest_row = filtered_df.sort_values("datetime").iloc[-1]
avg_error = filtered_df["Absolute Error MW"].mean()
mape = (
    abs(filtered_df["Actual Demand MW"] - filtered_df["Predicted Demand MW"])
    / filtered_df["Actual Demand MW"]
).mean() * 100

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
            f"{int(edges[i])}-{int(edges[i+1])} MW"
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
