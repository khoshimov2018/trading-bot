import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import sqlite3
import warnings
warnings.filterwarnings('ignore')

def walk_forward_validation(raw_df, train_years=3, step_months=6, min_test_days=60):

    results = []
    
    # Convert to datetime index if not already
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raw_df.index = pd.to_datetime(raw_df.index)
    
    start_date = raw_df.index.min()
    end_date = raw_df.index.max()
    
    # Calculate windows
    train_delta = timedelta(days=train_years * 365)
    step_delta = timedelta(days=step_months * 30)
    test_delta = timedelta(days=min_test_days)
    
    current_start = start_date
    
    while current_start + train_delta + test_delta <= end_date:
        train_end = current_start + train_delta
        test_start = train_end
        test_end = min(test_start + test_delta, end_date)
        
        # Extract windows
        train_mask = (raw_df.index >= current_start) & (raw_df.index < train_end)
        test_mask = (raw_df.index >= test_start) & (raw_df.index < test_end)
        
        if train_mask.sum() < 252 or test_mask.sum() < 20:  # Skip if insufficient data
            current_start += step_delta
            continue
            
        train_data = raw_df[train_mask]
        test_data = raw_df[test_mask]
        
        results.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_data': train_data,
            'test_data': test_data
        })
        
        current_start += step_delta
    
    return results

def add_transaction_costs(strategy_returns, cost_bps=2.0):

    # Assume daily rebalancing = daily round-trip
    cost_per_trade = cost_bps / 10000  # Convert bps to decimal
    
    # Simple approximation: subtract cost from each non-zero return
    adjusted_returns = strategy_returns.copy()
    trading_days = (strategy_returns != 0)
    adjusted_returns[trading_days] -= cost_per_trade
    
    return adjusted_returns

def calibrate_probabilities(train_probs, train_labels, test_probs):

    if len(np.unique(train_labels)) < 2:
        return test_probs  # Can't calibrate with single class
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_probs, train_labels)
    calibrated_probs = calibrator.transform(test_probs)
    
    return calibrated_probs

def compute_confusion_metrics(y_true, y_pred, threshold=0.5):

    y_pred_binary = (y_pred >= threshold).astype(int)
    
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_binary)) < 2:
        return {
            'confusion_matrix': np.zeros((2, 2)),
            'precision': 0.0,
            'recall': 0.0,
            'hit_rate': y_true.mean()
        }
    
    cm = confusion_matrix(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    hit_rate = (y_true == y_pred_binary).mean()
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'hit_rate': hit_rate
    }

def save_results_to_db(results, db_path='results.db'):
    """
    Save experiment results to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    
    # Create table if not exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            latent_dim INTEGER,
            window INTEGER,
            dropout REAL,
            l2_reg REAL,
            sharpe REAL,
            ic REAL,
            max_drawdown REAL,
            precision_score REAL,
            recall_score REAL,
            hit_rate REAL,
            test_start TEXT,
            test_end TEXT
        )
    ''')
    
    # Insert results
    for result in results:
        conn.execute('''
            INSERT INTO experiments 
            (timestamp, latent_dim, window, dropout, l2_reg, sharpe, ic, max_drawdown, 
             precision_score, recall_score, hit_rate, test_start, test_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            result.get('latent_dim', 0),
            result.get('window', 0),
            result.get('dropout', 0.0),
            result.get('l2_reg', 0.0),
            result.get('sharpe', 0.0),
            result.get('ic', 0.0),
            result.get('max_drawdown', 0.0),
            result.get('precision', 0.0),
            result.get('recall', 0.0),
            result.get('hit_rate', 0.0),
            result.get('test_start', ''),
            result.get('test_end', '')
        ))
    
    conn.commit()
    conn.close() 