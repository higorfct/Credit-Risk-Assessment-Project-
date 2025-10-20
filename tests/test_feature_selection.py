from src.feature_selection import select_features
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_select_features(sample_df):
    X, y, _ = encode_and_scale(sample_df)
    X_selected, selector, feature_names = select_features(X, y)
    assert X_selected.shape[1] == 8
    assert isinstance(feature_names, np.ndarray)
