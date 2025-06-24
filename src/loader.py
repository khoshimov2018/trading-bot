import pandas as pd

def load_sp500(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.set_index('Date').sort_index()
    return df[['Close/Last']].rename(columns={'Close/Last': 'Close'})

def load_vix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['DATE'], utc=True)
    df = df.set_index('Date').sort_index()
    return df[['CLOSE']].rename(columns={'CLOSE': 'VIX'}) 