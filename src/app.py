# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load the full customer summary data
summary_df = pd.read_excel("../Outputs/final_customer_summary_with_predictions.xlsx")

# Load churn and CLV prediction test sets 
churn_pred_df = pd.read_csv("../Outputs/churn_predictions_with_customer_id.csv")
clv_predictions_df = pd.read_csv("../Outputs/CLV_predictions_with_segments.csv")

# ===============================
# Section 1: Descriptive Analytics (Refined)
# ===============================

st.header("📊 Descriptive Analytics Summary")

# Business KPIs
st.subheader("🔹 Key Business Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", summary_df.shape[0])
col2.metric("Avg Customer Lifetime Value", f"€{summary_df['CLV'].mean():,.2f}")
col3.metric("Avg Recency (Days)", f"{summary_df['Recency'].mean():.0f}")
col4.metric("Avg Purchase Frequency", f"{summary_df['Frequency'].mean():.1f}")

# Pie Chart – Customer Segments
st.subheader("🔹 Customer Segmentation Overview")
fig_segment = px.pie(
    summary_df, 
    names='Segment_Name', 
    title='Customer Segments Based on RFM Analysis',
    hole=0.4, 
    template='plotly_white'
)
fig_segment.update_traces(textinfo='percent+label')
st.plotly_chart(fig_segment, use_container_width=True)

# Histogram – Churn Status
st.subheader("🔹 Churn Distribution")
# Replace 0 → 'Non-Churn', 1 → 'Churn' for display
summary_df['ChurnLabel'] = summary_df['ChurnFlag'].map({0: 'Non-Churn', 1: 'Churn'})

fig_churn = px.histogram(
    summary_df, 
    x='ChurnLabel', 
    color='ChurnLabel', 
    title='Customer Churn vs Non-Churn',
    template='plotly_white',
    color_discrete_map={'Churn': 'crimson', 'Non-Churn': 'seagreen'}
)
fig_churn.update_layout(xaxis_title="Churn Status", yaxis_title="Number of Customers", showlegend=False)
st.plotly_chart(fig_churn, use_container_width=True)

# Scatter – Recency vs Frequency by Segment
st.subheader("🔹 Purchase Behavior Insights")
fig_rfm = px.scatter(
    summary_df, 
    x='Recency', 
    y='Frequency', 
    size='MonetaryValue', 
    color='Segment_Name', 
    title='Customer Recency vs Frequency by Segment',
    template='plotly_white',
    size_max=50
)
fig_rfm.update_layout(xaxis_title="Recency (Days)", yaxis_title="Purchase Frequency")
st.plotly_chart(fig_rfm, use_container_width=True)