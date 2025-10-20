import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import fetch_data_from_db, replace_nulls, correct_typo_errors, handle_outliers
import const

def load_and_clean_data():
    df = fetch_data_from_db(const.sql_query)
    df['age'] = df['age'].astype(int)
    df['requested_value'] = df['requested_value'].astype(float)
    df['total_asset_value'] = df['total_asset_value'].astype(float)
    replace_nulls(df)
    return df

def preprocess_categorical(df):
    profession_translation = {
        'Advogado': 'Lawyer',
        'Arquiteto': 'Architect',
        'Cientista de Dados': 'Data Scientist',
        'Contador': 'Accountant',
        'Dentista': 'Dentist',
        'Empresário': 'Entrepreneur',
        'Engenheiro': 'Engineer',
        'Médico': 'Doctor',
        'Programador': 'Programmer'
    }
    df['profession'] = df['profession'].replace(profession_translation)
    valid_professions = list(profession_translation.values())
    correct_typo_errors(df, 'profession', valid_professions)
    return df

def handle_outlier_features(df):
    df = handle_outliers(df, 'years_in_profession', 0, 70)
    df = handle_outliers(df, 'age', 0, 110)
    df['requested_to_total_ratio'] = df['requested_value'] / df['total_asset_value']
    df['requested_to_total_ratio'] = df['requested_to_total_ratio'].astype(float)
    return df

def encode_and_scale(df):
    X = df.drop('class', axis=1)
    y = df['class'].map({'bad': 0, 'good': 1}).values

    features_num = [
        'years_in_profession', 'income', 'age', 'dependents',
        'requested_value', 'total_asset_value', 'requested_to_total_ratio'
    ]
    scaler = StandardScaler()
    X[features_num] = scaler.fit_transform(X[features_num])

    features_cat = ['profession', 'residence_type', 'education', 'score', 'marital_status', 'product']
    for col in features_cat:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y, scaler
