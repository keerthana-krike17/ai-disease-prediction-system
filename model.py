import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Load dataset
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, columns=["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"])
df.drop(columns=["RestingECG"], errors="ignore", inplace=True)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
features = X.columns.tolist()

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_scaled, y)

# SHAP Explainer
explainer = shap.Explainer(model)
shap_explainer = explainer(X_scaled)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(features, open("features.pkl", "wb"))
pickle.dump(explainer, open("shap_explainer.pkl", "wb"))
