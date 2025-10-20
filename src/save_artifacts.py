import joblib
import os

def save_artifacts(selector, scaler, model, path="./objects"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(selector, f"{path}/selector.joblib")
    joblib.dump(scaler, f"{path}/scaler.joblib")
    joblib.dump(model, f"{path}/RandomForest_model.joblib")
    print("Artifacts saved successfully in './objects/' folder")
