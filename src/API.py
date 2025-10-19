from flask import Flask, request, jsonify
import pandas as pd
import joblib
from utils import load_scalers, load_encoders

app = Flask(__name__)

# Load model and selector
model = joblib.load('./objects/RandomForest_model.joblib')
selector = joblib.load('./objects/selector.joblib')

# Features
numeric_columns = ['years_in_profession', 'income', 'age', 'dependents',
                   'requested_value', 'total_asset_value', 'requested_to_total_ratio']
categorical_columns = ['profession', 'residence_type', 'education', 'score', 
                       'marital_status', 'product']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame(input_data)

        # Apply scalers and encoders
        df = load_scalers(df, numeric_columns)
        df = load_encoders(df, categorical_columns)

        # Apply feature selector
        df_selected = selector.transform(df)

        # Make predictions
        pred_probs = model.predict_proba(df_selected)[:, 1]
        pred_classes = (pred_probs > 0.5).astype(int)

        return jsonify({
            'predicted_classes': pred_classes.tolist(),
            'predicted_probabilities': pred_probs.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
