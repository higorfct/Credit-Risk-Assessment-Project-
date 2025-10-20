import os
import joblib
import numpy as np
import pandas as pd
import random as python_random
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils import *
import const

# ==============================
# Set seed for reproducibility
# ==============================
seed = 41
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
python_random.seed(seed)

# ==============================
# Load and preprocess data
# ==============================
df = fetch_data_from_db(const.sql_query)

df['age'] = df['age'].astype(int)
df['requested_value'] = df['requested_value'].astype(float)
df['total_asset_value'] = df['total_asset_value'].astype(float)

replace_nulls(df)

profession_translation = {
    'Advogado': 'Lawyer',
    'Arquiteto': 'Architect',
    'Cientista de Dados': 'Data Scientist',
    'Contador': 'Accountant',
    'Dentista': 'Dentist',
    'Empresário': 'Entrepreneur',
    'Engenheiro': 'Engineer',
    'Médico': 'Doctor',
    'Programador': 'Programmer'
}
df['profession'] = df['profession'].replace(profession_translation)
valid_professions = list(profession_translation.values())
correct_typo_errors(df, 'profession', valid_professions)

df = handle_outliers(df, 'years_in_profession', 0, 70)
df = handle_outliers(df, 'age', 0, 110)

df['requested_to_total_ratio'] = df['requested_value'] / df['total_asset_value']
df['requested_to_total_ratio'] = df['requested_to_total_ratio'].astype(float)

# ==============================
# Prepare features and target
# ==============================
X = df.drop('class', axis=1)
y = df['class'].map({'bad': 0, 'good': 1}).values

features_num = [
    'years_in_profession', 'income', 'age', 'dependents',
    'requested_value', 'total_asset_value', 'requested_to_total_ratio'
]
scaler = StandardScaler()
X[features_num] = scaler.fit_transform(X[features_num])

features_cat = ['profession', 'residence_type', 'education', 'score', 'marital_status', 'product']
for col in features_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# ==============================
# Feature selection (RFE)
# ==============================
rf_for_rfe = RandomForestClassifier(
    n_estimators=10,
    max_depth=3,
    class_weight='balanced',
    random_state=seed
)
selector = RFE(rf_for_rfe, n_features_to_select=8, step=1)
selector = selector.fit(X, y)
X_selected = selector.transform(X)
feature_names = X.columns[selector.get_support()]

# ==============================
# Random Forest with Randomized Search
# ==============================
rf = RandomForestClassifier(class_weight='balanced', random_state=seed)

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring='roc_auc',
    cv=3,
    random_state=seed,
    n_jobs=-1
)
random_search.fit(X_selected, y)
rf_best = random_search.best_estimator_

print("Best Hyperparameters Found:")
print(random_search.best_params_)

# ==============================
# Leave-One-Out Cross-Validation
# ==============================
loo = LeaveOneOut()
y_true, y_pred, y_prob = [], [], []

for train_idx, test_idx in loo.split(X_selected):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    rf_best.set_params(class_weight=dict(enumerate(weights)))

    rf_best.fit(X_train, y_train)

    prob = rf_best.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)

    y_true.append(y_test[0])
    y_pred.append(pred[0])
    y_prob.append(prob[0])

# ==============================
# Evaluation
# ==============================
print("\n====== Random Forest Model Evaluation (LOO) ======")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
roc_auc = roc_auc_score(y_true, y_prob)
print(f"AUC-ROC: {roc_auc:.3f}")

# ==============================
# Feature Importance + Plot
# ==============================
importances = rf_best.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n====== Feature Importance ======")
print(feature_importance_df)

# Create output folder if not exists
os.makedirs('./objects', exist_ok=True)

# Save to CSV
feature_importance_path = './objects/feature_importance.csv'
feature_importance_df.to_csv(feature_importance_path, index=False)
print(f"Feature importance saved to '{feature_importance_path}'")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.tight_layout()

plot_path = './objects/feature_importance.png'
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Feature importance plot saved to '{plot_path}'")

# ==============================
# Save artifacts
# ==============================
joblib.dump(selector, './objects/selector.joblib')
joblib.dump(scaler, './objects/scaler.joblib')
joblib.dump(rf_best, './objects/RandomForest_model.joblib')
print("Selector, scaler and Random Forest model saved in './objects/' folder")

