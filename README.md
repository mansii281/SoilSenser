# 🌱 SoilSense: Explainable Soil Degradation & Health Analysis using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)](https://github.com/slundberg/shap)
[![Optuna](https://img.shields.io/badge/Optuna-Optimization-red)](https://optuna.org/)

---

## 🌟 Overview

**SoilSense** predicts **soil health and degradation** using a **hybrid ML model (XGBoost + Random Forest)** with **Optuna hyperparameter optimization**.

It calculates the **Soil Degradation Index (SDI)** and classifies soil as:

* 🟢 **Healthy**
* 🟡 **Low**
* 🟠 **Moderate**
* 🔴 **Highly Degraded**

Using **SHAP explainability**, SoilSense provides **actionable recommendations** for farmers on which soil parameters to adjust to improve soil health. Users can **download detailed reports**.

💡 **Real-Time Insights:** The dashboard updates predictions and recommendations in real-time when input parameters are adjusted, allowing farmers to instantly see the impact of changes on soil health.

---

## 🚀 Features

* 🔹 Hybrid ML Model: XGBoost + Random Forest
* 🔹 Optuna-based hyperparameter tuning
* 🔹 Predicts **Soil Degradation Index (SDI)** and class
* 🔹 **SHAP explainability** with actionable recommendations
* 🔹 Interactive **Streamlit dashboard** with **real-time updates**
* 🔹 Report generation and download
* 🔹 **Synthetic dataset creation** for training and evaluation

---

## 💡 Novelty Points

* 🌱 **Hybrid Model Approach:** Combines XGBoost and Random Forest for improved predictive performance and robustness.
* 🎯 **Optuna Optimization:** Automatically tunes hyperparameters for optimal model performance.
* 🔍 **Explainability with SHAP:** Provides farmers with actionable insights on which soil parameters to adjust.
* 📊 **Integrated Regression & Classification:** Predicts both Soil Degradation Index and discrete soil health classes.
* 💾 **Synthetic Dataset Generation:** Allows training and testing in the absence of extensive real-world soil datasets.
* 🖥️ **Interactive Dashboard:** User-friendly interface for real-time predictions and report downloads.
* ⚡ **Real-Time Parameter Adjustment:** Farmers can modify input parameters and immediately see changes in soil health predictions.

---

## 📝 Input Features

* 🌾 **N:** Nitrogen content
* 🌾 **P:** Phosphorus content
* 🌾 **K:** Potassium content
* 💧 **Moisture:** Soil moisture level
* 🌡️ **Temperature:** Ambient temperature
* 🌬️ **Humidity:** Ambient humidity

---

## 🤖 Machine Learning Approach

* 🌲 **Random Forest:** Captures non-linear relationships
* ⚡ **XGBoost:** Gradient boosting for high performance
* 🔗 **Hybrid Model:** Combines both for better accuracy
* 🎯 **Optuna Optimization:** Automatic hyperparameter tuning
* 🔍 **SHAP Explainer:** Provides interpretability and actionable recommendations

---

## 📊 Dataset

* File: `soil_dataset.csv`
* **Synthetic dataset** created for realistic soil health simulations

---

## 💾 Model Files

| 📁 File             | 📄 Description                     |
| ------------------- | ---------------------------------- |
| `rf_reg.pkl`        | Random Forest regression model     |
| `xgb_reg.pkl`       | XGBoost regression model           |
| `rf_cls.pkl`        | Random Forest classification model |
| `xgb_cls.pkl`       | XGBoost classification model       |
| `label_encoder.pkl` | Label encoder for soil classes     |

---

## 📈 Model Performance

### 🧮 Regression Metrics

| Metric | Train  | Test   |
| ------ | ------ | ------ |
| RMSE   | 0.7800 | 1.5766 |
| R²     | 0.9990 | 0.9960 |

### 🏷️ Classification Metrics

| Metric   | Train  | Test   |
| -------- | ------ | ------ |
| Accuracy | 99.94% | 96.25% |
| F1 Score | 0.9994 | 0.9622 |

### 🧩 Confusion Matrices

**Train Confusion Matrix:**

| Actual \ Pred | 🟢 Healthy | 🟡 Low | 🟠 Moderate | 🔴 High |
| ------------- | ---------- | ------ | ----------- | ------- |
| 🟢 Healthy    | 1614       | 0      | 0           | 0       |
| 🟡 Low        | 4          | 1602   | 0           | 0       |
| 🟠 Moderate   | 0          | 0      | 1588        | 0       |
| 🔴 High       | 0          | 0      | 0           | 1592    |

**Test Confusion Matrix:**

| Actual \ Pred | 🟢 Healthy | 🟡 Low | 🟠 Moderate | 🔴 High |
| ------------- | ---------- | ------ | ----------- | ------- |
| 🟢 Healthy    | 386        | 0      | 0           | 0       |
| 🟡 Low        | 12         | 373    | 9           | 0       |
| 🟠 Moderate   | 0          | 30     | 381         | 9       |
| 🔴 High       | 0          | 0      | 0           | 400     |

---

## 💻 Installation

```bash
git clone https://github.com/<your-username>/SoilSense.git
cd SoilSense
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Usage

1. Upload your soil dataset (`soil_dataset.csv`) with the input features.
2. Predict the **Soil Degradation Index** and soil class.
3. View **SHAP explanations** and actionable recommendations.
4. Adjust input parameters to see **real-time changes** in soil health predictions.
5. Download a **detailed report**.

---

## 📂 Folder Structure

```
SoilSense/
│
├── app.py                # Streamlit dashboard
├── dataset.py            # Dataset creation and preprocessing
├── model.py              # Model training and prediction functions
├── rf_cls.pkl            # Random Forest classification model
├── xgb_cls.pkl           # XGBoost classification model
├── rf_reg.pkl            # Random Forest regression model
├── xgb_reg.pkl           # XGBoost regression model
├── label_encoder.pkl     # Label encoder for soil classes
├── soil_dataset.csv      # Synthetic soil dataset
├── requirements.txt
└── soil_env/             # Virtual environment 
```

---

## 🔮 Future Enhancements

* 🌐 Integration with **IoT soil sensors** for real-time monitoring
* 🛰️ Incorporation of **satellite imagery** for spatial soil health analysis
* 📱 Deployment as a **web app** for farmers and agronomists

---
