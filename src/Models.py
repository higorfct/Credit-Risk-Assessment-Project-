import os
import joblib
import numpy as np
import pandas as pd
import random as python_random

from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import mlflow
import mlflow.sklearn

from utils import *
import const

# ==============================
# Set deterministic behavior for reproducibility
# ==============================
seed = 41
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
python_random.seed(seed)

# ==============================
# Initialize MLflow experiment
# ==============================
mlflow.set_experiment("Credit_Risk_Classification")
mlflow.start_run(run_name="LOO_Classification_Models")

# ==============================
# Load and preprocess dataset
# ==============================
df = fetch_data_from_db(const.sql_query)

# Ensure correct data types
df['age'] = df['age'].astype(int)
df['requested_value'] = df['requested_value'].astype(float)
df['total_asset_value'] = df['total_asset_value'].astype(float)

# Handle missing values
replace_nulls(df)

# Normalize profession names
profession_translation = {
    'Advogado': 'Lawyer', 'Arquiteto': 'Architect', 'Cientista de Dados': 'Data Scientist',
    'Contador': 'Accountant', 'Dentista': 'Dentist', 'Empresário': 'Entrepreneur',
    'Engenheiro': 'Engineer', 'Médico': 'Doctor', 'Programador': 'Programmer'
}
df['profession'] = df['profession'].replace(profession_translation)
valid_professions = list(profession_translation.values())
correct_typo_errors(df, 'profession', valid_professions)

# Handle outliers for numerical features
df = handle_outliers(df, 'years_in_profession', 0, 70)
df = handle_outliers(df, 'age', 0, 110)

# Feature engineering: ratio of requested value to total assets
df['requested_to_total_ratio'] = df['requested_value'] / df['total_asset_value']

# ==============================
# Prepare features and target
# ==============================
X = df.drop('class', axis=1)
y = df['class'].map({'bad': 0, 'good': 1}).values

# Scale numerical features
features_num = [
    'years_in_profession', 'income', 'age', 'dependents',
    'requested_value', 'total_asset_value', 'requested_to_total_ratio'
]
scaler = StandardScaler()
X[features_num] = scaler.fit_transform(X[features_num])

# Encode categorical features
features_cat = ['profession', 'residence_type', 'education', 'score', 'marital_status', 'product']
for col in features_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ==============================
# Feature selection using RFE with a small Random Forest
# ==============================
rf_for_rfe = RandomForestClassifier(n_estimators=10, max_depth=3, class_weight='balanced', random_state=seed)
selector = RFE(rf_for_rfe, n_features_to_select=8, step=1)
selector = selector.fit(X, y)
X_selected = selector.transform(X)
feature_names = X.columns[selector.get_support()]

# ==============================
# Define models and hyperparameter distributions
# ==============================
param_distributions = {
    "RandomForest": {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    },
    "SVM_RBF": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan']
    }
}

models = {
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=seed),
    "SVM_RBF": SVC(probability=True, class_weight='balanced', random_state=seed),
    "KNN": KNeighborsClassifier()
}

# ==============================
# Leave-One-Out Cross-Validation with Randomized Search
# ==============================
loo = LeaveOneOut()
results = {}

for name, model in models.items():
    print(f"\n===== Tuning and Training {name} =====")
    
    # Randomized Search CV (3-fold to reduce computation)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[name],
        n_iter=5,
        scoring='roc_auc',
        cv=3,
        random_state=seed,
        n_jobs=-1
    )
    random_search.fit(X_selected, y)
    best_model = random_search.best_estimator_
    print(f"Best hyperparameters for {name}: {random_search.best_params_}")

    y_true, y_pred, y_prob = [], [], []

    # Leave-One-Out predictions
    for train_idx, test_idx in loo.split(X_selected):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply class weights if model supports
        if hasattr(best_model, "set_params") and "class_weight" in best_model.get_params():
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            best_model.set_params(class_weight=dict(enumerate(weights)))

        best_model.fit(X_train, y_train)

        # Predict probabilities or classes
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(X_test)[:, 1]
        else:
            prob = best_model.predict(X_test)
        pred = (prob > 0.5).astype(int)

        y_true.append(y_test[0])
        y_pred.append(pred[0])
        y_prob.append(prob[0])

    # Store results
    results[name] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    # Log metrics and model to MLflow
    roc_auc = roc_auc_score(y_true, y_prob)
    mlflow.log_metric(f"{name}_ROC_AUC", roc_auc)
    mlflow.sklearn.log_model(best_model, f"{name}_model")

# ==============================
# Evaluation of all models
# ==============================
for name, res in results.items():
    print(f"\n====== {name} Model Evaluation (LOO) ======")
    print("\nConfusion Matrix:")
    print(confusion_matrix(res["y_true"], res["y_pred"]))
    print("\nClassification Report:")
    print(classification_report(res["y_true"], res["y_pred"]))
    roc_auc = roc_auc_score(res["y_true"], res["y_prob"])
    print(f"AUC-ROC: {roc_auc:.3f}")

# ==============================
# Save preprocessing artifacts
# ==============================
os.makedirs('./objects', exist_ok=True)
joblib.dump(selector, './objects/selector.joblib')
joblib.dump(scaler, './objects/scaler.joblib')
print("Selector and scaler saved in './objects/' folder")

mlflow.end_run()

