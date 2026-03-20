import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickel_files")
os.makedirs(PICKLE_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "Bank_Churn.csv"))

# Encode
le_geo = LabelEncoder()
le_gender = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Features for balance prediction
features = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
X = df[features]
y = df['Balance']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train GBR
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_scaled, y)

# Save
joblib.dump(gbr, os.path.join(PICKLE_DIR, "gbr.pkl"))
joblib.dump(scaler, os.path.join(PICKLE_DIR, "gbr_scaler.pkl"))
print("GBR model and scaler retrained successfully!")
