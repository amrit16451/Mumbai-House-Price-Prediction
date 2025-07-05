# 🏠 Mumbai House Price Prediction with XGBoost

This project focuses on building a machine learning model to predict property prices in Mumbai using housing data. The goal is to estimate the price of a house based on features like locality, type, area, age, and other relevant parameters.

The model uses XGBoost Regressor and handles both numerical and categorical data efficiently.

---

## 📊 Overview

The dataset contains property listings from Mumbai, including details like:

- **Type** of house (1 BHK, 2 BHK, etc.)
- **Locality** and **Region**
- **Status** (Ready to Move, Under Construction)
- **Area**,
- **Price**, with units in Lakhs or Crores

Prices are first converted to a uniform unit (₹ Crore), and categorical values are label encoded. After preprocessing, an XGBoost model is trained and evaluated.

---

## ✅ Features

- Price-unit conversion from Lakhs and Crores to ₹ Crores
- Label encoding for categorical variables
- Stratified train-test split based on locality
- XGBoost model with categorical support enabled
- Feature importance visualization
- Model performance metrics: R², RMSE, and MAE
- Reusable model and encoders saved with `joblib`

---

## 🔧 Technologies Used

- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn
- XGBoost
- Joblib

---

## 🧪 Model Performance

| Metric       | Value         |
|--------------|---------------|
| R² Score     | 0.93          |
| RMSE         | ₹0.58 Crore   |
| MAE          | ₹0.18 Crore   |

The model explains about 93% of the variance in the data, with an average prediction error of ₹18 Lakhs, which is strong for real estate pricing.

---

## 🔍 Feature Importance

The top contributing features to price prediction include:

- Property Area
- Locality
- Property Type
- Number of Bathrooms
- Age of Property

A visualization using `plot_importance()` is included in the project.

---

## 📦 Predicting on New Input

You can use the trained model to predict prices for new listings. Example:

```python
new_data = {
    'bhk': [2], 
    'type': ['Studio Apartment'],
    'locality': ['Godrej Urban Park'],
    'area': [1000], 
    'region': ['Powai'],
    'status': ['Ready to move'],
    'age': ['New'],         
}
predict_price(sample_input)  # Output: ₹1.43 Crore
