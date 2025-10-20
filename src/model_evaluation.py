import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, y_prob, model, feature_names, save_dir):
    print("\n====== Random Forest Model Evaluation (LOO) ======")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"AUC-ROC: {roc_auc:.3f}")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    os.makedirs(save_dir, exist_ok=True)
    feature_importance_path = os.path.join(save_dir, 'feature_importance.csv')
    feature_importance_df.to_csv(feature_importance_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
    plt.close()

    print("Feature importance and plot saved successfully.")
