import pandas as pd
from src.model_evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier
import os

def test_evaluate_model(tmp_path):
    X = [[0,1],[1,0]]
    y_true = [0,1]
    y_pred = [0,1]
    y_prob = [0.1, 0.9]
    model = RandomForestClassifier()
    model.n_features_in_ = 2
    model.feature_importances_ = [0.5,0.5]
    feature_names = ['f1','f2']
    
    evaluate_model(y_true, y_pred, y_prob, model, feature_names, save_dir=tmp_path)
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "feature_importance.png").exists()
