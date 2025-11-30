import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model & preprocessing
model = joblib.load("mumbai_house_model.pkl")
region_freq = joblib.load("region_freq.pkl")
locality_freq = joblib.load("locality_freq.pkl")

# Load dataset to build region → locality mapping
df = pd.read_csv("Mumbai House Prices.csv")

region_to_localities = (
    df.groupby("region")["locality"]
      .unique()
      .apply(sorted)
      .to_dict()
)

# All region list
all_regions = sorted(region_to_localities.keys())

st.title("🏠 Mumbai House Price Predictor")

# Inputs
bhk = st.number_input("BHK", 1, 10, 2)
area = st.number_input("Area (sq ft)", 100, 15000, 500)

region = st.selectbox("Select Region", all_regions)

# Locality dropdown dynamically updates
locality = st.selectbox(
    "Select Locality",
    region_to_localities.get(region, [])
)

age = st.selectbox("Age", ["Resale", "New", "Unknown"])
status = st.selectbox("Status", ["Ready to move", "Under Construction"])

# Encode
age_map = {'Resale': 0, 'New': 1, 'Unknown': 2}
status_map = {'Under Construction': 0, 'Ready to move': 1}

if st.button("Predict Price"):

    df_input = pd.DataFrame([{
        "bhk": bhk,
        "area": area,
        "pps": (1e7 / area),
        "age_encoded": age_map[age],
        "status_encoded": status_map[status],
        "region_freq": region_freq.get(region, 0),
        "locality_freq": locality_freq.get(locality, 0)
    }])

    pred_log = model.predict(df_input)[0]   # log prediction
    pred = np.expm1(pred_log)               # inverse log1p

    st.success(f"🏷 Estimated Price: **₹{pred:.2f} Cr**")
