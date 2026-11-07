import streamlit as st
import joblib
import numpy as np

# Load the real dataset model
model = joblib.load('kmeans_model.pkl')

st.title("📊 Real Customer Segmentation App")
st.write("Enter customer attributes to predict their marketing segment:")

# Input fields matching the real dataset
age = st.number_input("Customer Age", min_value=18, max_value=90, value=50)
income = st.number_input("Annual Income ($)", min_value=1000, max_value=200000, value=45000, step=1000)
total_spending = st.number_input("Total Spending across products ($)", min_value=0, max_value=5000, value=500, step=50)

# Cluster names based on your real dataset analysis
cluster_names = {
    0: "Budget Customers (Low Income, Low Spending)",
    1: "Silver Tier (Moderate Income, Good Spending)",
    2: "Platinum / VIP (High Income, High Spending)",
    3: "Conservative Shoppers (Middle Income, Low Spending)"
}

if st.button("Predict Segment"):
    features = np.array([[age, income, total_spending]])
    cluster = int(model.predict(features)[0])
    segment_name = cluster_names.get(cluster, f"Cluster {cluster}")
    
    st.success(f"Customer belongs to: **{segment_name}**")