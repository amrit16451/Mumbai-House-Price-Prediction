import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load trained model and dataset
# -----------------------
model = joblib.load("mumbai_house_price_model_latest.pkl")
feature_order = joblib.load("feature_order.pkl")
df = pd.read_csv("Mumbai House Prices.csv")   
# -----------------------
# Prepare option lists
# -----------------------
regions = sorted(df['region'].dropna().unique())
bhk_options = sorted(df['bhk'].dropna().unique())
property_types = sorted(df['type'].dropna().unique())
statuses = sorted(df['status'].dropna().unique())
ages = sorted(df['age'].dropna().unique())

# -----------------------
# Streamlit UI
# -----------------------
st.title("🏡 Real Estate Price Prediction App")
st.markdown("### Enter the property details below to get the estimated price (in Crores).")

# 1️⃣ Select Region
region = st.selectbox("Select Region", regions)

# 2️⃣ Locality depends on region
locality_list = sorted(df[df['region'] == region]['locality'].dropna().unique())
locality = st.selectbox("Select Locality", locality_list)

# 3️⃣ Other inputs
house_type = st.selectbox("Property Type", property_types)
bhk = st.selectbox("BHK", bhk_options)
status = st.selectbox("Status", statuses)
age = st.selectbox("Age of Property", ages)
area = st.number_input("Area (in sq.ft)", min_value=200, max_value=10000, value=1000, step=50)

# -----------------------
# Prepare input dataframe
# -----------------------
input_data = pd.DataFrame({
    'area': [area],
    'type': [house_type],
    'locality': [locality],
    'region': [region],
    'status': [status],
    'age': [age],
    'bhk': [bhk]
})

for col in ['type', 'locality', 'region', 'status', 'age', 'bhk']:
    input_data[col] = input_data[col].astype('category')

# -----------------------
# Prediction
# -----------------------
if st.button("🔮 Predict Price"):
    try:
        input_data = input_data[feature_order]
        price = model.predict(input_data)[0]
        st.success(f"🏠 Estimated Price: ₹ {price:.2f} Crores")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")