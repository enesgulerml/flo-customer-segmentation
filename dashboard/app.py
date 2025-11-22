import streamlit as st
import requests
import json

# API Configuration (Running on Docker port 8005)
API_URL = "http://127.0.0.1:8005/predict"

# Page Config
st.set_page_config(
    page_title="FLO Customer Segmentation",
    page_icon="👟",
    layout="centered"
)

# Header
st.title("👟 FLO Customer Segmentation")
st.markdown(
    """
    Enter customer metrics below to predict their segment using the 
    **K-Means** model served via **FastAPI**.
    """
)
st.divider()

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    total_orders = st.number_input(
        "Total Orders (Frequency)",
        min_value=1,
        value=5,
        help="Total number of online and offline orders."
    )
    total_price = st.number_input(
        "Total Spend (Monetary)",
        min_value=1.0,
        value=1500.50,
        format="%.2f",
        help="Total spending amount in local currency."
    )

with col2:
    recency = st.number_input(
        "Days Since Last Purchase (Recency)",
        min_value=0,
        value=30
    )
    tenure = st.number_input(
        "Days Since First Purchase (Tenure)",
        min_value=0,
        value=500
    )

st.divider()

# --- Prediction Logic ---
if st.button("🔍 Identify Segment", type="primary", use_container_width=True):

    # Prepare payload matching app/schema.py
    payload = {
        "recency_days": recency,
        "total_orders": total_orders,
        "total_price": total_price,
        "tenure_days": tenure
    }

    try:
        # Call the API
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            segment_name = result["cluster_name"]
            cluster_id = result["cluster_id"]
            model_version = result["model_version"]

            # Display Result
            st.success(f"**Customer Segment:** {segment_name} (Cluster {cluster_id})")
            st.caption(f"Prediction served by model version: {model_version}")

            # Business Recommendations (Static Logic)
            if cluster_id == 0:  # Example logic
                st.info("📢 **Action:** Send a 'We Miss You' coupon.")
            elif cluster_id == 1:
                st.info("⭐ **Action:** Offer early access to new collections (VIP).")
            elif cluster_id == 2:
                st.info("✅ **Action:** Enroll in loyalty program.")
            elif cluster_id == 3:
                st.warning("⚠️ **Action:** High risk of churn. Recommend discount.")

        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"Connection Error: Is the API running on port 8005? \n\nDetails: {e}")