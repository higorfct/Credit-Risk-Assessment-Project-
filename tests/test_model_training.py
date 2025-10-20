from src.model_training import train_random_forest, cross_validate_loo
from sklearn.datasets import make_classification
import numpy as np

def test_train_random_forest():
    X, y = make_classification(n_samples=50, n_features=8, random_state=42)
    model = train_random_forest(X, y, feature_names=[f"f{i}" for i in range(8)])
    assert hasattr(model, "predict")

def test_cross_validate_loo():
    X, y = make_classification(n_samples=10, n_features=8, random_state=42)
    model = train_random_forest(X, y, feature_names=[f"f{i}" for i in range(8)])
    y_true, y_pred, y_prob = cross_validate_loo(model, X, y)
    assert len(y_true) == len(y)
    assert len(y_pred) == len(y)
