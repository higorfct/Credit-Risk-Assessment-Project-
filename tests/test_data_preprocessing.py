from src.data_processing import replace_nulls, correct_typo_errors, handle_outliers, encode_and_scale

def test_replace_nulls(sample_df):
    sample_df.loc[0, 'income'] = None
    df_clean = replace_nulls(sample_df)
    assert df_clean['income'].isnull().sum() == 0

def test_correct_typo_errors(sample_df):
    df = correct_typo_errors(sample_df, 'profession', ['Lawyer', 'Engineer', 'Doctor'])
    assert set(df['profession'].unique()).issubset({'Lawyer', 'Engineer', 'Doctor', 'Unknown'})

def test_handle_outliers(sample_df):
    df = handle_outliers(sample_df, 'age', 20, 50)
    assert df['age'].max() <= 50 and df['age'].min() >= 20

def test_encode_and_scale(sample_df):
    X, y, scaler = encode_and_scale(sample_df)
    assert X.shape[0] == sample_df.shape[0]
    assert len(y) == sample_df.shape[0]
