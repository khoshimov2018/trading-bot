import pandas as pd
import numpy as np

def predict_prob(lstm, X, dates):

    p = lstm.predict(X, verbose=0).flatten()
    return pd.Series(p, index=dates, name='prob').to_frame()

def apply_rule(prob_df: pd.DataFrame, ret_series: pd.Series, threshold=0.5, 
               position_sizing='binary', vol_target=0.10):

    fwd_ret = ret_series.shift(-1)  # next day's return
    
    if position_sizing == 'binary':
        # Original binary rule: 1=long, 0=cash
        signals = (prob_df['prob'] >= threshold).astype(int)
    
    elif position_sizing == 'prob_weighted':
        # Position = 2 * (p - 0.5), clipped to ±1
        signals = 2 * (prob_df['prob'] - 0.5)
        signals = signals.clip(-1, 1)
    
    elif position_sizing == 'vol_adjusted':
        # Vol-adjusted position sizing
        signals = 2 * (prob_df['prob'] - 0.5)
        signals = signals.clip(-1, 1)
        
        # Estimate 20-day rolling volatility
        vol_est = ret_series.rolling(20).std() * np.sqrt(252)  # Annualized
        vol_est = vol_est.fillna(vol_est.mean())  # Fill NaN with mean
        
        # Adjust position by vol target
        vol_adj = vol_target / vol_est
        signals = signals * vol_adj.shift(1)  # Use previous day's vol estimate
        signals = signals.clip(-1, 1)  # Keep within reasonable bounds
    
    else:
        raise ValueError(f"Unknown position_sizing: {position_sizing}")
    
    strat_ret = signals * fwd_ret
    
    out = pd.DataFrame({
        'signal': signals,
        'strat_ret': strat_ret,
        'bench_ret': fwd_ret  # buy-and-hold for comparison
    })
    return out.dropna() 