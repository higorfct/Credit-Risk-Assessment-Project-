import os

SEED = 41
os.environ['PYTHONHASHSEED'] = str(SEED)

OBJECTS_PATH = "./objects/"
FEATURE_IMPORTANCE_PATH = os.path.join(OBJECTS_PATH, "feature_importance.csv")
MODEL_PATH = os.path.join(OBJECTS_PATH, "RandomForest_model.joblib")
SCALER_PATH = os.path.join(OBJECTS_PATH, "scaler.joblib")
SELECTOR_PATH = os.path.join(OBJECTS_PATH, "selector.joblib")
PLOT_PATH = os.path.join(OBJECTS_PATH, "feature_importance.png")
