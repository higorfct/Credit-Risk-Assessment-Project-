from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def select_features(X, y, seed=41):
    rf = RandomForestClassifier(n_estimators=10, max_depth=3, class_weight='balanced', random_state=seed)
    selector = RFE(rf, n_features_to_select=8, step=1)
    selector.fit(X, y)
    X_selected = selector.transform(X)
    feature_names = X.columns[selector.get_support()]
    return X_selected, selector, feature_names
