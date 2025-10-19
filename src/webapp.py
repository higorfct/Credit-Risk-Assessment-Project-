import streamlit as st
import requests
import yaml

# Page config
st.set_page_config(
    page_title="Credit Evaluation",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Evaluation Application")
st.markdown("#### Assess your client's credit risk quickly and visually")

# Load API URL
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    url = config["url_api"]["url"]

# Options
professions = ['Lawyer', 'Architect', 'Data Scientist', 'Accountant', 'Dentist', 
               'Entrepreneur', 'Engineer', 'Doctor', 'Programmer']
residence_types = ['Rented', 'Other', 'Owned']
education_levels = ['Elementary', 'HighSchool', 'PostGraduate', 'Higher']
scores = ['Low', 'Fair', 'Good', 'VeryGood']
marital_statuses = ['Married', 'Divorced', 'Single', 'Widowed']
products = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 
            'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

# Form
with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        profession = st.selectbox("Profession", professions)
        years_in_profession = st.number_input("Years in profession", min_value=0, value=0)
        income = st.number_input("Monthly income", min_value=0.0, value=0.0)
        residence_type = st.selectbox("Residence type", residence_types)
        education = st.selectbox("Education level", education_levels)
        score = st.selectbox("Score", scores)
    with col2:
        age = st.number_input("Age", min_value=18, max_value=110, value=25)
        dependents = st.number_input("Dependents", min_value=0, value=0)
        marital_status = st.selectbox("Marital status", marital_statuses)
        product = st.selectbox("Product", products)
        requested_value = st.number_input("Requested amount", min_value=0.0, value=0.0)
        total_asset_value = st.number_input("Total asset value", min_value=0.0, value=0.0)

    submit_button = st.form_submit_button(label="🔍 Predict Credit Risk")

# Send to API
if submit_button:
    requested_to_total_ratio = requested_value / total_asset_value if total_asset_value > 0 else 0

    new_data = {
        'profession': [profession],
        'years_in_profession': [years_in_profession],
        'income': [income],
        'residence_type': [residence_type],
        'education': [education],
        'score': [score],
        'age': [age],
        'dependents': [dependents],
        'marital_status': [marital_status],
        'product': [product],
        'requested_value': [requested_value],
        'total_asset_value': [total_asset_value],
        'requested_to_total_ratio': [requested_to_total_ratio]
    }

    try:
        response = requests.post(url, json=new_data)
        if response.status_code == 200:
            result = response.json()
            predicted_class = result.get("predicted_classes", [None])[0]
            prob = result.get("predicted_probabilities", [None])[0] * 100

            # Display result
            if prob >= 70:
                color = "#4CAF50"
                message = "Excellent credit profile — low default risk."
            elif 40 <= prob < 70:
                color = "#FFC107"
                message = "Moderate credit profile — consider manual review."
            else:
                color = "#F44336"
                message = "High risk — credit approval not recommended."

            st.markdown(f"""
                <div style="background-color:{color};padding:16px;border-radius:10px;">
                    <h3 style="color:white;">Credit Risk Result</h3>
                    <p><strong>Predicted Probability of Good Credit:</strong> {prob:.2f}%</p>
                    <p><strong>Predicted Class:</strong> {"Good" if predicted_class == 1 else "Bad"}</p>
                    <p>{message}</p>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Error sending data to API: {e}")
