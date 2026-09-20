# 🩺 Diabetes Prediction & Visual Analytics System

An end-to-end Machine Learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict diabetes risk using the PIMA Diabetes Dataset.

---

## 📌 Features

- **Interactive Diagnostic Portal:** Real-time prediction interface with adjustable patient clinical parameters.
- **Exploratory Data Analysis (EDA):** Visual insights including correlation heatmaps, class distribution charts, and kernel density estimation (KDE) distributions.
- **Model Explainability:** Evaluation metrics detailing training/testing accuracy, confusion matrix, and feature weight coefficients of the linear SVM model.

---

## 🛠️ Tech Stack

- **Machine Learning:** Scikit-Learn (Support Vector Machines, StandardScaler)
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Seaborn, Matplotlib
- **Web Interface:** Streamlit

---

## 🚀 Quickstart Guide

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/diabetes-prediction-app.git](https://github.com/your-username/diabetes-prediction-app.git)
   cd diabetes-prediction-app
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Web Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Performance

- **Training Accuracy:** ~78.6%
- **Test Accuracy:** ~77.2%
- **Algorithm:** Support Vector Classifier (Linear Kernel)