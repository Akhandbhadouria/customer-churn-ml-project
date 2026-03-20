# 🏦 ChurnVision: Bank Customer Churn Analytics Dashboard

**ChurnVision** is a premium, end-to-end Machine Learning web application designed to predict and analyze customer attrition for banking institutions. It combines multiple classification models, regression models, and clustering algorithms into a single, interactive dashboard to provide actionable insights into customer behavior.

---

### 🌐 [Live Demo Link](https://customer-churn-ml-project.onrender.com/)

---

## 🚀 Key Features

### 1. **Multi-Model Churn Prediction**
Predicts whether a customer is likely to leave the bank using an ensemble of high-performance models:
*   **XGBoost Classifier**: State-of-the-art gradient boosting with **~87% accuracy**.
*   **KNN (K-Nearest Neighbors)**: Similarity-based classification.
*   **Decision Tree**: Interpretable logic-based classification.

### 2. **Financial Estimation**
*   **Gradient Boosting Regressor**: Dynamically estimates a customer's expected account balance based on their financial profile (Credit Score, Tenure, Products, etc.).

### 3. **Behavioral Segmentation**
*   **K-Means Clustering**: Automatically segments customers into distinct behavioral profiles using optimized PCA and the "Elbow Method."

### 4. **Bulk Analysis Tool**
*   Upload a CSV file to perform mass predictions and generate a summary report for thousands of customers in seconds.

### 5. **Premium Interactive UI**
*   Modern, responsive dashboard with glassmorphism design.
*   Real-time probability visualizations and performance benchmarking charts.

---

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, Gunicorn
*   **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
*   **Frontend**: HTML5, CSS3 (Vanilla), JavaScript, Chart.js
*   **Deployment**: Ready for Render, Heroku, or AWS (includes `Procfile` and `requirements.txt`)

---

## 📦 Project Structure

```bash
├── app.py              # Main Flask Application
├── Bank_Churn.csv      # Primary Dataset
├── pickel_files/       # Serialized ML Models & Scalers
├── static/             # CSS & Client-side Assets
├── templates/          # HTML Templates
├── train_xgb.py        # XGBoost Training Script
├── fix_gbr.py          # Model version compatibility fix script
├── requirements.txt    # Project Dependencies
└── Procfile            # Deployment Configuration
```

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Akhandbhadouria/customer-churn-ml-project.git
    cd customer-churn-ml-project
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Locally**:
    ```bash
    python app.py
    ```
    Access the dashboard at `http://127.0.0.1:5001`.

---

## 📊 Deployment

This project is configured for one-click deployment to **Render** or **Heroku**:
*   **Build Command**: `pip install -r requirements.txt`
*   **Start Command**: `gunicorn app:app`

---

## 📝 Authors
*   **Akhand Bhadouria**

---
