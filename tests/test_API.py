import requests
import yaml

# Load API URL from config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    url = config['url_api']['url']

# New input data (English column names)
new_data = [
    {
        'profession': 'Lawyer',
        'years_in_profession': 39,
        'income': 20860.0,
        'residence_type': 'Rented',
        'education': 'Elementary',
        'score': 'Low',
        'age': 36,
        'dependents': 0,
        'marital_status': 'Widowed',
        'product': 'DoubleDuty',
        'requested_value': 139244.0,
        'total_asset_value': 320000.0,
        'requested_to_total_ratio': 2.2
    }
]

# Send POST request
response = requests.post(url, json=new_data)

if response.status_code == 200:
    print("Predictions received:")
    predictions = response.json()
    print(predictions)
else:
    print("Error during prediction:", response.status_code)
    print(response.text)
