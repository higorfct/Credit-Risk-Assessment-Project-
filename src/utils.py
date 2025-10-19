from fuzzywuzzy import process
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml
import psycopg2
import const
import joblib
import os

def fetch_data_from_db(sql_query):
    """
    Fetch data from a PostgreSQL database using credentials stored in config.yaml.
    Returns the result as a pandas DataFrame.
    """
    try:
        # Load database configuration from YAML file
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Establish connection to PostgreSQL
        con = psycopg2.connect(
            dbname=config['database_config']['dbname'],
            user=config['database_config']['user'],
            password=config['database_config']['password'],
            host=config['database_config']['host']
        )

        # Execute SQL query
        cursor = con.cursor()
        cursor.execute(sql_query)

        # Fetch all rows and convert them into a DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:
        # Ensure the connection and cursor are closed properly
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    return df


def replace_nulls(df):
    """
    Replace missing values in a DataFrame:
      - For categorical columns: replace with mode (most frequent value)
      - For numeric columns: replace with median
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
        else:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)


def correct_typo_errors(df, column, valid_list):
    """
    Correct spelling mistakes or inconsistent values in a specific column using fuzzy matching.
    - df: input DataFrame
    - column: column name to correct
    - valid_list: list of valid (expected) values
    """
    for i, value in enumerate(df[column]):
        value_str = str(value) if pd.notnull(value) else value

        # If the value is not in the valid list, try to correct it using fuzzy matching
        if value_str not in valid_list and pd.notnull(value_str):
            correction = process.extractOne(value_str, valid_list)[0]
            df.at[i, column] = correction


def handle_outliers(df, column, min_value, max_value):
    """
    Replace outliers with the median of valid values within the specified range.
    """
    median_value = df[(df[column] >= min_value) & (df[column] <= max_value)][column].median()
    df[column] = df[column].apply(lambda x: median_value if x < min_value or x > max_value else x)
    return df


def save_scalers(df, column_names):
    """
    Standardize numeric columns and save the fitted scalers as joblib files.
    Each column will have its own scaler saved in './objects/'.
    """
    for column_name in column_names:
        scaler = StandardScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
        joblib.dump(scaler, f"./objects/scaler_{column_name}.joblib")

    return df


def save_encoders(df, column_names):
    """
    Encode categorical columns using LabelEncoder and save each encoder as a joblib file.
    Each column will have its own encoder saved in './objects/'.
    """
    for column_name in column_names:
        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])
        joblib.dump(label_encoder, f"./objects/labelencoder_{column_name}.joblib")

    return df

def load_scalers(df, column_names):
    """
    Applies saved scalers (.joblib files) to numerical columns in the DataFrame.
    """
    for column_name in column_names:
        scale_file_name = f"./objects/scaler_{column_name}.joblib"
        scaler = joblib.load(scale_file_name)
        # Ensure the result stays as a Series
        df[column_name] = scaler.transform(df[[column_name]]).flatten()
    return df


def safe_transform(value, encoder):
    """
    Transforms a value using a LabelEncoder, returning None if the value was unseen during training.
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return None  # or some default value you prefer


def load_encoders(df, column_names):
    """
    Applies saved LabelEncoders (.joblib files) to categorical columns in the DataFrame.
    """
    for column_name in column_names:
        encoder_file_name = f"./objects/labelencoder_{column_name}.joblib"
        label_encoder = joblib.load(encoder_file_name)
        df[column_name] = df[column_name].apply(lambda x: safe_transform(x, label_encoder))
    return df
