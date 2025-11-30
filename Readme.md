# 🏠 Mumbai House Price Prediction with CAT-Boost

This project focuses on building a machine learning model to predict property prices in Mumbai using housing data. The goal is to estimate the price of a house based on features like locality, type, area, age, and other relevant parameters.

The model uses CAT-Boost Regressor and handles both numerical and categorical data efficiently.

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
- Feature importance visualization
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
| RMSE         | ₹0.15 Crore   |

---

## 🔍 Feature Importance

The top contributing features to price prediction include:

- Property Area
- Locality
- Property Type
- Number of Bathrooms
- Age of Property


A visualization using `plot_importance()` is included in the project.
