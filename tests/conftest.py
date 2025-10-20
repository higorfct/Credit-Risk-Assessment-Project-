import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    """DataFrame de exemplo para testes"""
    data = {
        'age': [25, 40, 60],
        'years_in_profession': [2, 15, 35],
        'income': [3000, 5000, 8000],
        'dependents': [0, 2, 3],
        'requested_value': [10000, 20000, 30000],
        'total_asset_value': [50000, 150000, 200000],
        'profession': ['Lawyer', 'Engineer', 'Doctor'],
        'residence_type': ['Owned', 'Rented', 'Owned'],
        'education': ['Bachelor', 'Master', 'PhD'],
        'score': ['A', 'B', 'C'],
        'marital_status': ['Single', 'Married', 'Married'],
        'product': ['Car', 'Home', 'Car'],
        'class': ['good', 'bad', 'good']
    }
    return pd.DataFrame(data)
