#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import modules
from src.loader import load_sp500, load_vix
from src.features import make_daily_features, train_val_test, make_latent_series, to_sequences
from src.autoencoder import train_autoencoder
from src.lstm_model import train_lstm
from src.backtest import predict_prob, apply_rule
from src.metrics import sharpe, max_drawdown, information_coefficient
from src.validation import (walk_forward_validation, add_transaction_costs, 
                           calibrate_probabilities, compute_confusion_metrics, 
                           save_results_to_db)

def run_single_experiment(train_data, test_data, latent_dim, window, 
                         dropout=0.0, l2_reg=0.0, position_sizing='binary',
                         ae_epochs=100, lstm_epochs=50):
    """Run a single experiment with given parameters."""
    try:
        # Split train data into train/val
        n_train = int(len(train_data) * 0.85)  # 85% train, 15% val from training period
        actual_train = train_data.iloc[:n_train]
        actual_val = train_data.iloc[n_train:]
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        train_scaled = pd.DataFrame(
            scaler.fit_transform(actual_train), 
            index=actual_train.index, 
            columns=actual_train.columns)
        val_scaled = pd.DataFrame(
            scaler.transform(actual_val), 
            index=actual_val.index, 
            columns=actual_val.columns)
        test_scaled = pd.DataFrame(
            scaler.transform(test_data), 
            index=test_data.index, 
            columns=test_data.columns)
        
        # Train autoencoder (exclude RET from features)
        train_features = train_scaled.drop('RET', axis=1)
        val_features = val_scaled.drop('RET', axis=1)
        test_features = test_scaled.drop('RET', axis=1)
        
        encoder = train_autoencoder(train_features, val_features, 
                                  latent_dim=latent_dim, epochs=ae_epochs)
        
        # Create latent representations
        train_lat = make_latent_series(encoder, train_features)
        val_lat = make_latent_series(encoder, val_features)
        test_lat = make_latent_series(encoder, test_features)
        
        # Add back RET for labels
        train_lat['RET'] = train_data.loc[train_lat.index, 'RET']
        val_lat['RET'] = train_data.loc[val_lat.index, 'RET']
        test_lat['RET'] = test_data.loc[test_lat.index, 'RET']
        
        # Create sequences
        X_tr, y_tr, d_tr = to_sequences(train_lat, window=window)
        X_va, y_va, d_va = to_sequences(val_lat, window=window)
        X_te, y_te, d_te = to_sequences(test_lat, window=window)
        
        if len(X_tr) < 10 or len(X_va) < 5 or len(X_te) < 5:
            return None  # Insufficient data
        
        # Train LSTM
        lstm = train_lstm(X_tr, y_tr, X_va, y_va, 
                         window=window, latent_dim=latent_dim,
                         epochs=lstm_epochs, dropout=dropout, l2_reg=l2_reg)
        
        # Predictions on validation (for calibration)
        val_prob_df = predict_prob(lstm, X_va, d_va)
        val_probs = val_prob_df['prob'].values
        
        # Predictions on test
        test_prob_df = predict_prob(lstm, X_te, d_te)
        test_probs = test_prob_df['prob'].values
        
        # Calibrate probabilities
        calibrated_probs = calibrate_probabilities(val_probs, y_va, test_probs)
        test_prob_df['prob'] = calibrated_probs
        
        # Backtest with different position sizing
        results = {}
        for pos_sizing in ['binary', 'prob_weighted', 'vol_adjusted']:
            bt_result = apply_rule(test_prob_df, test_data['RET'], 
                                 position_sizing=pos_sizing)
            
            # Add transaction costs
            bt_result['strat_ret_tc'] = add_transaction_costs(bt_result['strat_ret'])
            
            # Calculate metrics
            cum_ret = (1 + bt_result['strat_ret']).cumprod()
            cum_ret_tc = (1 + bt_result['strat_ret_tc']).cumprod()
            
            sr = sharpe(bt_result['strat_ret'])
            sr_tc = sharpe(bt_result['strat_ret_tc'])
            mdd = max_drawdown(cum_ret)
            mdd_tc = max_drawdown(cum_ret_tc)
            ic = information_coefficient(calibrated_probs, bt_result['bench_ret'])
            
            # Confusion matrix metrics
            confusion_metrics = compute_confusion_metrics(y_te, calibrated_probs)
            
            results[pos_sizing] = {
                'sharpe': sr,
                'sharpe_tc': sr_tc,
                'max_drawdown': mdd,
                'max_drawdown_tc': mdd_tc,
                'ic': ic,
                'precision': confusion_metrics['precision'],
                'recall': confusion_metrics['recall'],
                'hit_rate': confusion_metrics['hit_rate'],
                'confusion_matrix': confusion_metrics['confusion_matrix']
            }
        
        return results
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        return None

def main():
    print("Starting comprehensive robustness testing...")
    
    # Load data
    print("Loading data...")
    sp = load_sp500('data/HistoricalData_1750587928127.csv')
    vix = load_vix('data/VIX_History.csv')
    raw_df = make_daily_features(sp, vix)
    
    print(f"Data loaded: {len(raw_df)} samples from {raw_df.index.min()} to {raw_df.index.max()}")
    
    # Walk-forward validation setup
    print("Setting up walk-forward validation...")
    validation_windows = walk_forward_validation(raw_df, train_years=3, step_months=6)
    print(f"Generated {len(validation_windows)} validation windows")
    
    # Hyperparameter grids
    window_grid = [10, 15, 25]  # Task R1
    latent_grid = [4, 8, 12]    # Task R2
    dropout_grid = [0.0, 0.2]   # Task R3
    l2_grid = [0.0, 1e-4]       # Task R3
    
    all_results = []
    total_experiments = len(validation_windows) * len(window_grid) * len(latent_grid) * len(dropout_grid) * len(l2_grid)
    experiment_count = 0
    
    print(f"Total experiments to run: {total_experiments}")
    
    # Run experiments
    for window_idx, (latent_dim, window, dropout, l2_reg) in enumerate(
        itertools.product(latent_grid, window_grid, dropout_grid, l2_grid)):
        
        print(f"\nHyperparameters: latent_dim={latent_dim}, window={window}, dropout={dropout}, l2_reg={l2_reg}")
        
        window_results = []
        
        for val_window in validation_windows:
            experiment_count += 1
            print(f"  Experiment {experiment_count}/{total_experiments}: "
                  f"Train {val_window['train_start'].strftime('%Y-%m')} to "
                  f"{val_window['train_end'].strftime('%Y-%m')}, "
                  f"Test {val_window['test_start'].strftime('%Y-%m')} to "
                  f"{val_window['test_end'].strftime('%Y-%m')}")
            
            result = run_single_experiment(
                val_window['train_data'], val_window['test_data'],
                latent_dim, window, dropout, l2_reg,
                ae_epochs=50, lstm_epochs=30  # Reduced for speed
            )
            
            if result is not None:
                # Store results for each position sizing method
                for pos_sizing, metrics in result.items():
                    result_record = {
                        'latent_dim': latent_dim,
                        'window': window,
                        'dropout': dropout,
                        'l2_reg': l2_reg,
                        'position_sizing': pos_sizing,
                        'test_start': val_window['test_start'].strftime('%Y-%m-%d'),
                        'test_end': val_window['test_end'].strftime('%Y-%m-%d'),
                        **metrics
                    }
                    window_results.append(result_record)
                    all_results.append(result_record)
        
        # Print aggregated results for this hyperparameter combination
        if window_results:
            df_results = pd.DataFrame(window_results)
            print(f"  Results across {len(window_results)} experiments:")
            for pos_sizing in ['binary', 'prob_weighted', 'vol_adjusted']:
                subset = df_results[df_results['position_sizing'] == pos_sizing]
                if len(subset) > 0:
                    print(f"    {pos_sizing}:")
                    print(f"      Sharpe (no TC): {subset['sharpe'].mean():.3f} ± {subset['sharpe'].std():.3f}")
                    print(f"      Sharpe (w/ TC):  {subset['sharpe_tc'].mean():.3f} ± {subset['sharpe_tc'].std():.3f}")
                    print(f"      Hit Rate:       {subset['hit_rate'].mean():.3f} ± {subset['hit_rate'].std():.3f}")
    
    # Save all results to database
    print(f"\nSaving {len(all_results)} results to database...")
    save_results_to_db(all_results)
    
    # Final analysis
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*60)
        
        # Best configurations by position sizing
        for pos_sizing in ['binary', 'prob_weighted', 'vol_adjusted']:
            subset = df_all[df_all['position_sizing'] == pos_sizing]
            if len(subset) > 0:
                best_sharpe = subset.loc[subset['sharpe_tc'].idxmax()]
                best_hit_rate = subset.loc[subset['hit_rate'].idxmax()]
                
                print(f"\n{pos_sizing.upper()} POSITION SIZING:")
                print(f"  Best Sharpe (w/ TC): {best_sharpe['sharpe_tc']:.3f}")
                print(f"    Config: latent_dim={best_sharpe['latent_dim']}, window={best_sharpe['window']}, "
                      f"dropout={best_sharpe['dropout']}, l2_reg={best_sharpe['l2_reg']}")
                print(f"    Hit Rate: {best_sharpe['hit_rate']:.3f}")
                print(f"    Max Drawdown: {best_sharpe['max_drawdown_tc']:.3f}")
                
                print(f"  Best Hit Rate: {best_hit_rate['hit_rate']:.3f}")
                print(f"    Config: latent_dim={best_hit_rate['latent_dim']}, window={best_hit_rate['window']}, "
                      f"dropout={best_hit_rate['dropout']}, l2_reg={best_hit_rate['l2_reg']}")
                print(f"    Sharpe (w/ TC): {best_hit_rate['sharpe_tc']:.3f}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total experiments: {len(df_all)}")
        print(f"  Mean Sharpe (w/ TC): {df_all['sharpe_tc'].mean():.3f}")
        print(f"  Std Sharpe (w/ TC):  {df_all['sharpe_tc'].std():.3f}")
        print(f"  Mean Hit Rate:       {df_all['hit_rate'].mean():.3f}")
        print(f"  Profitable configs:  {(df_all['sharpe_tc'] > 0).mean():.1%}")
        
    print(f"\nRobustness testing complete! Results saved to results.db")

if __name__ == "__main__":
    main() 