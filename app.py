"""
Bank Customer Churn Analysis - Web Application
A modern Flask web application for predicting customer churn using ML models.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ─── Setup and Directories ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickel_files")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the dataset for statistics
df = pd.read_csv(os.path.join(BASE_DIR, "Bank_Churn.csv"))

# Load KNN model
knn_model = joblib.load(os.path.join(PICKLE_DIR, "knn.pkl"))
knn_scaler = joblib.load(os.path.join(PICKLE_DIR, "knn_scaler.pkl"))

# Load Decision Tree model
dt_model = joblib.load(os.path.join(PICKLE_DIR, "dt.pkl"))
dt_scaler = joblib.load(os.path.join(PICKLE_DIR, "dt_scaler.pkl"))

# Load XGBoost model
xgb_model = joblib.load(os.path.join(PICKLE_DIR, "xgb.pkl"))
xgb_scaler = joblib.load(os.path.join(PICKLE_DIR, "xgb_scaler.pkl"))

# Load Gradient Boosting Regressor model (may fail on sklearn version mismatch)
try:
    gbr_model = joblib.load(os.path.join(PICKLE_DIR, "gbr.pkl"))
    gbr_scaler = joblib.load(os.path.join(PICKLE_DIR, "gbr_scaler.pkl"))
    GBR_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not load GBR model: {e}")
    gbr_model = None
    gbr_scaler = None
    GBR_AVAILABLE = False

# Load K-Means model
kmeans_model = joblib.load(os.path.join(PICKLE_DIR, "kmeans.pkl"))
kmeans_scaler = joblib.load(os.path.join(PICKLE_DIR, "kmeans_scaler.pkl"))

# Load label encoders
le_gender = joblib.load(os.path.join(PICKLE_DIR, "le_gender.pkl"))
le_geo = joblib.load(os.path.join(PICKLE_DIR, "le_geo.pkl"))

# Average balance for replacement
avg_balance = joblib.load(os.path.join(PICKLE_DIR, "avg_balance.pkl"))


def get_dataset_stats():
    """Compute dashboard statistics from the dataset."""
    total = len(df)
    churned = int(df["Exited"].sum())
    not_churned = total - churned
    churn_rate = round(churned / total * 100, 1)

    geo_counts = df["Geography"].value_counts().to_dict()
    gender_counts = df["Gender"].value_counts().to_dict()

    # Churn by geography
    churn_by_geo = df.groupby("Geography")["Exited"].mean().round(3).to_dict()

    # Churn by gender
    churn_by_gender = df.groupby("Gender")["Exited"].mean().round(3).to_dict()

    # Age distribution stats
    age_stats = {
        "mean": round(float(df["Age"].mean()), 1),
        "min": int(df["Age"].min()),
        "max": int(df["Age"].max()),
    }

    # Credit score stats
    credit_stats = {
        "mean": round(float(df["CreditScore"].mean()), 1),
        "min": int(df["CreditScore"].min()),
        "max": int(df["CreditScore"].max()),
    }

    # Balance stats
    balance_stats = {
        "mean": round(float(df["Balance"].mean()), 2),
        "min": round(float(df["Balance"].min()), 2),
        "max": round(float(df["Balance"].max()), 2),
    }

    # Salary stats
    salary_stats = {
        "mean": round(float(df["EstimatedSalary"].mean()), 2),
        "min": round(float(df["EstimatedSalary"].min()), 2),
        "max": round(float(df["EstimatedSalary"].max()), 2),
    }

    # Churn by number of products
    churn_by_products = df.groupby("NumOfProducts")["Exited"].mean().round(3).to_dict()
    churn_by_products = {str(k): v for k, v in churn_by_products.items()}

    # Churn by active member
    churn_by_active = df.groupby("IsActiveMember")["Exited"].mean().round(3).to_dict()
    churn_by_active = {str(k): v for k, v in churn_by_active.items()}

    # Age bins for distribution
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)
    age_distribution = df["AgeGroup"].value_counts().sort_index().to_dict()
    age_distribution = {str(k): int(v) for k, v in age_distribution.items()}
    churn_by_age_group = df.groupby("AgeGroup")["Exited"].mean().round(3).to_dict()
    churn_by_age_group = {str(k): float(v) for k, v in churn_by_age_group.items()}

    return {
        "total_customers": total,
        "churned": churned,
        "not_churned": not_churned,
        "churn_rate": churn_rate,
        "geo_counts": geo_counts,
        "gender_counts": gender_counts,
        "churn_by_geo": churn_by_geo,
        "churn_by_gender": churn_by_gender,
        "age_stats": age_stats,
        "credit_stats": credit_stats,
        "balance_stats": balance_stats,
        "salary_stats": salary_stats,
        "churn_by_products": churn_by_products,
        "churn_by_active": churn_by_active,
        "age_distribution": age_distribution,
        "churn_by_age_group": churn_by_age_group,
    }


@app.route("/")
def index():
    stats = get_dataset_stats()
    return render_template("index.html", stats=json.dumps(stats))


@app.route("/api/predict/churn", methods=["POST"])
def predict_churn():
    """Predict churn using KNN and Decision Tree models."""
    try:
        data = request.get_json()

        credit_score = float(data["credit_score"])
        geography = data["geography"]
        gender = data["gender"]
        age = float(data["age"])
        tenure = float(data["tenure"])
        balance = float(data["balance"])
        num_products = float(data["num_products"])
        has_card = float(data["has_card"])
        is_active = float(data["is_active"])
        salary = float(data["salary"])

        # Encode categorical features
        geo_enc = le_geo.transform([geography])[0]
        gen_enc = le_gender.transform([gender])[0]

        # Replace 0 balance with average
        if balance == 0:
            balance = avg_balance

        # Prepare feature vector
        features = np.array([[credit_score, geo_enc, gen_enc, age, tenure, balance,
                              num_products, has_card, is_active, salary]])

        col_names = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        features_df = pd.DataFrame(features, columns=col_names)

        # KNN prediction
        knn_scaled = knn_scaler.transform(features_df)
        knn_pred = int(knn_model.predict(knn_scaled)[0])
        knn_proba = knn_model.predict_proba(knn_scaled)[0].tolist()

        # Decision Tree prediction
        dt_scaled = dt_scaler.transform(features_df)
        dt_pred = int(dt_model.predict(dt_scaled)[0])
        dt_proba = dt_model.predict_proba(dt_scaled)[0].tolist()

        # XGBoost prediction
        xgb_scaled = xgb_scaler.transform(features_df)
        xgb_pred = int(xgb_model.predict(xgb_scaled)[0])
        xgb_proba = xgb_model.predict_proba(xgb_scaled)[0].tolist()

        return jsonify({
            "success": True,
            "knn": {
                "prediction": "Churn" if knn_pred == 1 else "Not Churn",
                "churn_probability": round(knn_proba[1] * 100, 1),
                "stay_probability": round(knn_proba[0] * 100, 1),
            },
            "decision_tree": {
                "prediction": "Churn" if dt_pred == 1 else "Not Churn",
                "churn_probability": round(dt_proba[1] * 100, 1),
                "stay_probability": round(dt_proba[0] * 100, 1),
            },
            "xgboost": {
                "prediction": "Churn" if xgb_pred == 1 else "Not Churn",
                "churn_probability": round(xgb_proba[1] * 100, 1),
                "stay_probability": round(xgb_proba[0] * 100, 1),
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict/balance", methods=["POST"])
def predict_balance():
    """Predict estimated balance using Gradient Boosting Regressor."""
    try:
        if not GBR_AVAILABLE:
            # Fallback: return the dataset average balance
            return jsonify({
                "success": True,
                "predicted_balance": round(float(avg_balance), 2),
                "note": "GBR model unavailable; using dataset average."
            })

        data = request.get_json()

        credit_score = float(data["credit_score"])
        geography = data["geography"]
        gender = data["gender"]
        age = float(data["age"])
        tenure = float(data["tenure"])
        num_products = float(data["num_products"])
        has_card = float(data["has_card"])
        is_active = float(data["is_active"])
        salary = float(data["salary"])

        geo_enc = le_geo.transform([geography])[0]
        gen_enc = le_gender.transform([gender])[0]

        features = np.array([[credit_score, geo_enc, gen_enc, age, tenure,
                              num_products, has_card, is_active, salary]])

        col_names = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        features_df = pd.DataFrame(features, columns=col_names)

        gbr_scaled = gbr_scaler.transform(features_df)
        predicted_balance = float(gbr_model.predict(gbr_scaled)[0])

        return jsonify({
            "success": True,
            "predicted_balance": round(predicted_balance, 2),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict/cluster", methods=["POST"])
def predict_cluster():
    """Assign customer to a cluster using K-Means."""
    try:
        data = request.get_json()

        credit_score = float(data["credit_score"])
        age = float(data["age"])
        tenure = float(data["tenure"])
        balance = float(data["balance"])
        num_products = float(data["num_products"])
        salary = float(data["salary"])
        has_card = float(data["has_card"])
        is_active = float(data["is_active"])

        features = np.array([[credit_score, age, tenure, balance, num_products, salary, has_card, is_active]])

        col_names = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "HasCrCard", "IsActiveMember"]
        features_df = pd.DataFrame(features, columns=col_names)

        km_scaled = kmeans_scaler.transform(features_df)
        cluster = int(kmeans_model.predict(km_scaled)[0])

        # Cluster descriptions
        cluster_labels = {
            0: "Standard Customer",
            1: "High-Value Customer",
            2: "At-Risk Customer",
            3: "New/Low-Engagement Customer",
        }

        cluster_desc = {
            0: "Average credit score, moderate engagement, and balanced account activity.",
            1: "Premium segment with high balance and strong credit score. Priority retention target.",
            2: "Shows signs of disengagement. May benefit from targeted retention campaigns.",
            3: "Newer customer with lower tenure and balance. Growth opportunity segment.",
        }

        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        description = cluster_desc.get(cluster, "No description available for this cluster.")

        return jsonify({
            "success": True,
            "cluster": cluster,
            "label": label,
            "description": description,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/stats")
def api_stats():
    """Return dataset statistics as JSON."""
    return jsonify(get_dataset_stats())


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle CSV file upload, perform analysis and bulk prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"})
        
        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Only CSV files are allowed"})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and analyze the uploaded file
        uploaded_df = pd.read_csv(filepath)
        
        required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                         "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        
        # Check if all columns match (case insensitive check)
        missing_cols = [c for c in required_cols if c not in uploaded_df.columns]
        if missing_cols:
            return jsonify({"success": False, "error": f"Missing columns in CSV: {', '.join(missing_cols)}"})

        # Analysis logic
        total_rows = len(uploaded_df)
        
        # Preprocessing for predictions
        df_pred = uploaded_df.copy()
        
        # Encode geography and gender
        df_pred['Geography_enc'] = le_geo.transform(df_pred['Geography'])
        df_pred['Gender_enc'] = le_gender.transform(df_pred['Gender'])
        
        # Replace 0 balance with project average balance
        df_pred['Balance'] = df_pred['Balance'].replace(0, avg_balance)
        
        # Prepare feature vector (all models use same columns ORDER)
        col_names = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        
        # Remap encoded columns to standard names for scaler
        X_df = df_pred.copy()
        X_df['Geography'] = df_pred['Geography_enc']
        X_df['Gender'] = df_pred['Gender_enc']
        X_df = X_df[col_names]

        # Bulk Predictions
        # KNN
        X_knn = knn_scaler.transform(X_df)
        knn_preds = knn_model.predict(X_knn)
        knn_churn_count = int(np.sum(knn_preds))
        
        # Decision Tree
        X_dt = dt_scaler.transform(X_df)
        dt_preds = dt_model.predict(X_dt)
        dt_churn_count = int(np.sum(dt_preds))
        
        # XGBoost
        X_xgb = xgb_scaler.transform(X_df)
        xgb_preds = xgb_model.predict(X_xgb).astype(int)
        xgb_churn_count = int(np.sum(xgb_preds))

        # Perform basic statistics on uploaded file
        stats = {
            "total_customers": total_rows,
            "geo_counts": uploaded_df["Geography"].value_counts().to_dict(),
            "gender_counts": uploaded_df["Gender"].value_counts().to_dict(),
            "age_stats": {
                "mean": round(float(uploaded_df["Age"].mean()), 1),
                "min": int(uploaded_df["Age"].min()),
                "max": int(uploaded_df["Age"].max()),
            },
            "balance_stats": {
                "mean": round(float(uploaded_df["Balance"].mean()), 2),
                "min": round(float(uploaded_df["Balance"].min()), 2),
                "max": round(float(uploaded_df["Balance"].max()), 2),
            }
        }

        # If file has actual results (Exited), include accuracy report
        actual_churn = None
        if "Exited" in uploaded_df.columns:
            actual_churn = int(uploaded_df["Exited"].sum())
            stats["actual_churn_rate"] = round(actual_churn / total_rows * 100, 1)

        return jsonify({
            "success": True,
            "filename": filename,
            "total_rows": total_rows,
            "predictions": {
                "knn": {"churned": knn_churn_count, "rate": round(knn_churn_count / total_rows * 100, 1)},
                "decision_tree": {"churned": dt_churn_count, "rate": round(dt_churn_count / total_rows * 100, 1)},
                "xgboost": {"churned": xgb_churn_count, "rate": round(xgb_churn_count / total_rows * 100, 1)}
            },
            "actual_churn": actual_churn,
            "stats": stats
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Get port from environment variable for service like Heroku, Render, etc.
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
