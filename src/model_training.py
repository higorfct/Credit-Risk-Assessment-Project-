import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_random_forest(X, y, feature_names, seed=41):
    rf = RandomForestClassifier(class_weight='balanced', random_state=seed)
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=10,
        scoring='roc_auc', cv=3, random_state=seed, n_jobs=-1
    )
    random_search.fit(X, y)
    rf_best = random_search.best_estimator_

    print("Best Hyperparameters Found:")
    print(random_search.best_params_)
    return rf_best

def cross_validate_loo(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        model.set_params(class_weight=dict(enumerate(weights)))

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob > 0.5).astype(int)

        y_true.append(y_test[0])
        y_pred.append(pred[0])
        y_prob.append(prob[0])

    return y_true, y_pred, y_prob
