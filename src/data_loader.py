import pandas as pd

def load_news_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def load_stock_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df
