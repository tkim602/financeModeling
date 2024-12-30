import pandas as pd
from config import DATA_PATH, DATE_COL

def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, utc=True)
    df.sort_values(by=DATE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
