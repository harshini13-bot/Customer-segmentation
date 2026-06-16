import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------
# Page Config
# --------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Segmentation Dashboard")
st.markdown("---")

# --------------------------
# Load Model
# --------------------------

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/clustered_customers.csv")

recommendations = {

    "Premium Customers":
    "VIP Membership, Luxury Products",

    "Budget Customers":
    "Discounts, Coupons, Cashback",

    "Occasional Buyers":
    "Festival Offers, Loyalty Rewards",

    "Young High Spenders":
    "Fashion, Electronics, Lifestyle Products",

    "Loyal Customers":
    "Membership Plans, Exclusive Deals"
}

# --------------------------
# Sidebar
# --------------------------

st.sidebar.header("Customer Details")

age = st.sidebar.slider(
    "Age",
    18,
    70,
    30
)

income = st.sidebar.slider(
    "Annual Income (k$)",
    10,
    150,
    50
)

spending = st.sidebar.slider(
    "Spending Score",
    1,
    100,
    50
)

# --------------------------
# Predict Cluster
# --------------------------

if st.sidebar.button("Predict Customer Segment"):

    sample = pd.DataFrame({
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    segment_names = {
        0: "Premium Customers",
        1: "Budget Customers",
        2: "Occasional Buyers",
        3: "Young High Spenders",
        4: "Loyal Customers"
    }

    segment = segment_names[prediction]

    st.success(f"Predicted Segment : {segment}")
    st.info(recommendations[segment])

# --------------------------
# KPI Cards
# --------------------------

st.subheader("Project KPIs")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Customers",
    len(df)
)

c2.metric(
    "Clusters",
    df["Cluster"].nunique()
)

c3.metric(
    "Average Income",
    round(df["Annual Income (k$)"].mean(), 2)
)

c4.metric(
    "Average Spending",
    round(df["Spending Score (1-100)"].mean(), 2)
)

st.markdown("---")
# --------------------------
# Search Customer
# --------------------------
st.markdown("---")

st.subheader("🔍 Search Customer")

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=1,
    max_value=int(df["CustomerID"].max()),
    value=1
)

if st.button("Search"):

    customer = df[
        df["CustomerID"] == customer_id
    ]

    st.dataframe(customer)
# --------------------------
# Dataset Preview
# --------------------------

st.subheader("Dataset Preview")

st.dataframe(df)

# --------------------------
# Pie Chart
# --------------------------
st.subheader("🥧 Segment Distribution")

fig_pie = px.pie(
    df,
    names="Segment",
    title="Customer Segments"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)
# --------------------------
# Bar Chart
# --------------------------
st.subheader("📊 Segment Count")

segment_count = (
    df["Segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = [
    "Segment",
    "Count"
]

fig_bar = px.bar(
    segment_count,
    x="Segment",
    y="Count"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# --------------------------
# Cluster Distribution
# --------------------------

st.subheader("Cluster Distribution")

fig = px.histogram(
    df,
    x="Cluster",
    color="Cluster"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------
# Income vs Spending
# --------------------------

st.subheader("Income vs Spending")

fig2 = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    hover_data=["Age"]
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------
# Age Distribution
# --------------------------

st.subheader("Age Distribution")

fig3 = px.histogram(
    df,
    x="Age"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# --------------------------
# 3D Visualization
# --------------------------

st.subheader("🚀 3D Customer Segmentation")

fig3d = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Segment"
)

st.plotly_chart(
    fig3d,
    use_container_width=True
)

# --------------------------
# Cluster Summary
# --------------------------

st.subheader("Cluster Summary")

summary = df.groupby("Cluster").agg({

    "Age": "mean",

    "Annual Income (k$)": "mean",

    "Spending Score (1-100)": "mean"

}).round(2)

st.dataframe(summary)

st.subheader("📈 Segment Analytics")

segment_summary = (
    df.groupby("Segment")
    .agg({
        "Age":"mean",
        "Annual Income (k$)":"mean",
        "Spending Score (1-100)":"mean",
        "CustomerID":"count"
    })
    .rename(columns={
        "CustomerID":"Customer Count"
    })
    .round(2)
)

st.dataframe(segment_summary)

# --------------------------
# Download CSV
# --------------------------

st.subheader("📥 Download Report")

csv = df.to_csv(index=False)

st.download_button(
    "Download CSV Report",
    csv,
    "customer_segmentation_report.csv",
    "text/csv"
)