import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler

def make_daily_features(sp: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    # Align indices
    df = sp.join(vix, how='inner')

    # 1. Daily log-return
    df['RET'] = np.log(df['Close']).diff()

    # 2. VIX level (already present as 'VIX')
    # 3. 14-period RSI on price
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # Drop incomplete rows
    df = df.dropna()

    return df[['RET', 'VIX', 'RSI']]

def train_val_test(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train+n_val]
    test  = df.iloc[n_train+n_val:]

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train), index=train.index, columns=train.columns)
    val_scaled   = pd.DataFrame(
        scaler.transform(val), index=val.index, columns=val.columns)
    test_scaled  = pd.DataFrame(
        scaler.transform(test), index=test.index, columns=test.columns)

    return train_scaled, val_scaled, test_scaled, scaler

def make_latent_series(encoder, df: pd.DataFrame, prefix='Z'):
    latent = encoder.predict(df, verbose=0)
    cols   = [f'{prefix}{i+1}' for i in range(latent.shape[1])]
    latent_df = pd.DataFrame(latent, index=df.index, columns=cols)
    return latent_df 

def to_sequences(latent_df: pd.DataFrame, window: int = 10):
    """
    Returns X (samples, window, latent_dim) and y (next-day binary label).
    """
    X, y, dates = [], [], []
    # Separate latent features from RET column
    latent_cols = [col for col in latent_df.columns if col != 'RET']
    latent_features = latent_df[latent_cols]
    
    for i in range(window, len(latent_df)-1):
        X.append(latent_features.iloc[i-window:i].values)
        y.append(int(latent_df['RET'].iloc[i+1] > 0))  # RET for labels
        dates.append(latent_df.index[i])
    return np.array(X), np.array(y), np.array(dates) 