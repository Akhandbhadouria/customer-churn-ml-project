import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickel_files")
DATA_FILE = os.path.join(BASE_DIR, "Bank_Churn.csv")

# Ensure pickle directory exists
if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

# Load data
df = pd.read_csv(DATA_FILE)

# Preprocessing
if 'CustomerId' in df.columns:
    df = df.drop(['CustomerId'], axis=1)
if 'Surname' in df.columns:
    df = df.drop(['Surname'], axis=1)

# Categorical encoding (using same logic as app.py)
le_geo = LabelEncoder()
le_gender = LabelEncoder()

df['Geography'] = le_geo.fit_transform(df['Geography'])
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Handling balance (average replacement for 0 as in app.py)
avg_balance = df[df['Balance'] != 0]['Balance'].mean()
df['Balance'] = df['Balance'].replace(0, avg_balance)

# Features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (reproducible split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Accuracy check
train_acc = xgb_model.score(X_train, y_train)
test_acc = xgb_model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Save model and scaler
joblib.dump(xgb_model, os.path.join(PICKLE_DIR, "xgb.pkl"))
joblib.dump(scaler, os.path.join(PICKLE_DIR, "xgb_scaler.pkl"))

# Save encoders if they don't exist (though they should)
# joblib.dump(le_geo, os.path.join(PICKLE_DIR, "le_geo.pkl"))
# joblib.dump(le_gender, os.path.join(PICKLE_DIR, "le_gender.pkl"))

print("XGBoost model and scaler saved successfully.")
