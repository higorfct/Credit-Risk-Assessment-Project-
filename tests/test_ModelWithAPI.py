import pytest
from src.API import app
import joblib
import os

@pytest.fixture
def client(tmp_path):
    """Create a test client and save a fake model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    # Create a fake Random Forest model
    model = RandomForestClassifier()
    model.n_features_in_ = 2
    model.feature_importances_ = np.array([0.5, 0.5])
    
    # Save the model to the temporary path
    joblib.dump(model, tmp_path / "RandomForest_model.joblib")

    # Point the API to load artifacts from the temporary path
    os.environ["ARTIFACTS_PATH"] = str(tmp_path)

    # Yield the Flask test client
    with app.test_client() as client:
        yield client

def test_api_integration(client):
    """Test the /predict endpoint of the API."""
    # Sample input data
    data = {
        "age": 30,
        "years_in_profession": 5,
        "income": 4000,
        "dependents": 1,
        "requested_value": 10000,
        "total_asset_value": 50000,
        "profession": "Lawyer",
        "residence_type": "Owned",
        "education": "Bachelor",
        "score": "A",
        "marital_status": "Single",
        "product": "Car"
    }

    # Send POST request to /predict
    response = client.post("/predict", json=data)

    # Check response status
    assert response.status_code == 200

    # Check response content
    json_data = response.get_json()
    assert "prediction" in json_data
    assert "probability" in json_data
