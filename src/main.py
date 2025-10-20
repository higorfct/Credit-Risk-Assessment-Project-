from data_processing import load_and_clean_data, preprocess_categorical, handle_outlier_features, encode_and_scale
from feature_selection import select_features
from model_training import train_random_forest, cross_validate_loo
from model_evaluation import evaluate_model
from save_artifacts import save_artifacts
import config

def main():
    df = load_and_clean_data()
    df = preprocess_categorical(df)
    df = handle_outlier_features(df)
    X, y, scaler = encode_and_scale(df)
    X_selected, selector, feature_names = select_features(X, y, config.SEED)
    rf_model = train_random_forest(X_selected, y, feature_names, config.SEED)
    y_true, y_pred, y_prob = cross_validate_loo(rf_model, X_selected, y)
    evaluate_model(y_true, y_pred, y_prob, rf_model, feature_names, config.OBJECTS_PATH)
    save_artifacts(selector, scaler, rf_model, config.OBJECTS_PATH)

if __name__ == "__main__":
    main()
