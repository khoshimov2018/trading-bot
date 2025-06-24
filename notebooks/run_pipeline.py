#!/usr/bin/env python3


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- 1
from src.loader import load_sp500, load_vix
sp  = load_sp500('data/HistoricalData_1750587928127.csv')
vix = load_vix('data/VIX_History.csv')


# -- 2
from src.features import make_daily_features
raw_df = make_daily_features(sp, vix)


# -- 3
from src.features import train_val_test
train_df, val_df, test_df, scaler = train_val_test(raw_df)


# -- 4
from src.autoencoder import train_autoencoder
# Remove RET from scaled data for autoencoder training
train_features = train_df.drop('RET', axis=1)
val_features = val_df.drop('RET', axis=1)
test_features = test_df.drop('RET', axis=1)
encoder = train_autoencoder(train_features, val_features, latent_dim=2, epochs=200)


# -- 5  (append RET back in for labels)
from src.features import make_latent_series

train_lat = make_latent_series(encoder, train_features)
val_lat   = make_latent_series(encoder, val_features)
test_lat  = make_latent_series(encoder, test_features)

# Add back the unscaled RET for labels
train_lat['RET'] = raw_df.loc[train_lat.index, 'RET']
val_lat['RET'] = raw_df.loc[val_lat.index, 'RET']
test_lat['RET'] = raw_df.loc[test_lat.index, 'RET']


# -- 6
from src.features import to_sequences
win = 10
X_tr, y_tr, d_tr = to_sequences(train_lat, window=win)
X_va, y_va, d_va = to_sequences(val_lat,   window=win)
X_te, y_te, d_te = to_sequences(test_lat,  window=win)


# -- 7
from src.lstm_model import train_lstm
lstm = train_lstm(X_tr, y_tr, X_va, y_va, window=win, latent_dim=2,
                  epochs=100)


# -- 8-9
from src.backtest import predict_prob, apply_rule
prob_te = predict_prob(lstm, X_te, d_te)
bt_te   = apply_rule(prob_te, raw_df['RET'])


# -- 10
from src.metrics import sharpe, max_drawdown, information_coefficient
cum = (1 + bt_te['strat_ret']).cumprod()
sr  = sharpe(bt_te['strat_ret'])
mdd = max_drawdown(cum)
ic  = information_coefficient(prob_te['prob'], bt_te['bench_ret'])

print(f"Out-of-sample Sharpe: {sr:.2f}")
print(f"Max Drawdown:        {mdd:.2%}")
print(f"Information Coeff.:  {ic:.3f}")

# Task 12 - Hyperparameter Sweep (Optional)
# Search over latent_dim in {2,4,6} and window in {5,10,20} to find the best combination.

# Task 12 - Hyperparameter sweep
import itertools
import pandas as pd

def run_experiment(latent_dim, window):
    """Run full pipeline for given hyperparameters"""
    # 4. Train AE
    train_features = train_df.drop('RET', axis=1)
    val_features = val_df.drop('RET', axis=1)
    test_features = test_df.drop('RET', axis=1)
    encoder = train_autoencoder(train_features, val_features, latent_dim=latent_dim, epochs=200)
    
    # 5. Latent DF
    
    train_lat = make_latent_series(encoder, train_features)
    val_lat   = make_latent_series(encoder, val_features)
    test_lat  = make_latent_series(encoder, test_features)
    
    # Add back the unscaled RET for labels
    train_lat['RET'] = raw_df.loc[train_lat.index, 'RET']
    val_lat['RET'] = raw_df.loc[val_lat.index, 'RET']
    test_lat['RET'] = raw_df.loc[test_lat.index, 'RET']
    
    # 6. Sequences
    X_tr, y_tr, d_tr = to_sequences(train_lat, window=window)
    X_va, y_va, d_va = to_sequences(val_lat,   window=window)
    X_te, y_te, d_te = to_sequences(test_lat,  window=window)
    
    # 7. Train LSTM
    lstm = train_lstm(X_tr, y_tr, X_va, y_va, window=window, latent_dim=latent_dim, epochs=100)
    
    # 8-10. Predict & evaluate
    prob_te = predict_prob(lstm, X_te, d_te)
    bt_te   = apply_rule(prob_te, raw_df['RET'])
    
    sr  = sharpe(bt_te['strat_ret'])
    ic  = information_coefficient(prob_te['prob'], bt_te['bench_ret'])
    
    return sr, ic, encoder, lstm

# Grid search
latent_dims = [2, 4, 6]
windows = [5, 10, 20]
results = []

print("Running hyperparameter sweep...")
for latent_dim, window in itertools.product(latent_dims, windows):
    print(f"Testing latent_dim={latent_dim}, window={window}")
    sr, ic, enc, lstm_model = run_experiment(latent_dim, window)
    results.append({
        'latent_dim': latent_dim,
        'window': window,
        'sharpe': sr,
        'ic': ic,
        'encoder': enc,
        'lstm': lstm_model
    })
    print(f"  -> Sharpe: {sr:.3f}, IC: {ic:.3f}")

# Find best combination
results_df = pd.DataFrame(results)
best_idx = results_df['sharpe'].idxmax()
best_result = results[best_idx]

print(f"\nBest combination:")
print(f"latent_dim={best_result['latent_dim']}, window={best_result['window']}")
print(f"Sharpe: {best_result['sharpe']:.3f}, IC: {best_result['ic']:.3f}")

# Use best models for saving
best_encoder = best_result['encoder']
best_lstm = best_result['lstm']

# Task 13 - Model Serialization
# Save the best models and preprocessing artifacts.

# Task 13 - Save artifacts
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save best models (if hyperparameter sweep was run)
try:
    best_encoder.save('models/encoder.keras')
    best_lstm.save('models/lstm.keras')
    print("Saved best models from hyperparameter sweep")
except NameError:
    # If no sweep was run, save the single run models
    encoder.save('models/encoder.keras')
    lstm.save('models/lstm.keras')
    print("Saved models from single run")

# Save scaler
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
print("Saved scaler")

print("All artifacts saved to models/ directory")

