from src.utils import handle_outliers
import pandas as pd

def test_handle_outliers_utils():
    df = pd.DataFrame({'a':[1,100,5]})
    df = handle_outliers(df, 'a', 0, 50)
    assert df['a'].max() <= 50
