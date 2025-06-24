#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed. Install with: pip install yfinance")
    yf = None

from src.features import make_daily_features, make_latent_series, to_sequences

class LiveTrader:
    def __init__(self, model_dir='models'):
        """Initialize live trader with saved models."""
        self.model_dir = model_dir
        self.encoder = None
        self.lstm = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load saved models and scaler."""
        try:
            self.encoder = tf.keras.models.load_model(f'{self.model_dir}/encoder.keras')
            self.lstm = tf.keras.models.load_model(f'{self.model_dir}/lstm.keras')
            
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def fetch_market_data(self, lookback_days=60):
        """Fetch recent market data from Yahoo Finance."""
        if yf is None:
            raise ImportError("yfinance package required for live data")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # Fetch S&P 500 and VIX data
            sp_data = yf.download("^GSPC", start=start_date, end=end_date)[['Close']]
            vix_data = yf.download("^VIX", start=start_date, end=end_date)[['Close']]
            
            # Fix column structure to match training data format
            sp_data = sp_data.squeeze()  # Convert to Series if single column
            if isinstance(sp_data, pd.Series):
                sp_data = sp_data.to_frame('Close')
            
            vix_data = vix_data.squeeze()  # Convert to Series if single column
            if isinstance(vix_data, pd.Series):
                vix_data = vix_data.to_frame('VIX')
            else:
                vix_data = vix_data.rename(columns={'Close': 'VIX'})
            
            print(f"Fetched {len(sp_data)} days of S&P 500 data")
            print(f"Fetched {len(vix_data)} days of VIX data")
            
            return sp_data, vix_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            raise
    
    def prepare_features(self, sp_data, vix_data):
        """Prepare features for prediction."""
        try:
            # Create features using the same pipeline as training
            raw_features = make_daily_features(sp_data, vix_data)
            
            if len(raw_features) < 15:  # Need sufficient data for RSI calculation
                raise ValueError(f"Insufficient data: only {len(raw_features)} days available")
            
            # Scale all features (scaler was trained on RET, VIX, RSI)
            scaled_features = pd.DataFrame(
                self.scaler.transform(raw_features),
                index=raw_features.index,
                columns=raw_features.columns
            )
            
            # Extract only the features for autoencoder (VIX, RSI)
            features_for_encoder = scaled_features[['VIX', 'RSI']]
            
            print(f"Prepared {len(scaled_features)} days of features")
            return raw_features, features_for_encoder
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            raise
    
    def generate_signal(self, raw_features, scaled_features, window=10):
        """Generate trading signal for the next day."""
        try:
            if len(scaled_features) < window + 1:
                raise ValueError(f"Need at least {window + 1} days of data, got {len(scaled_features)}")
            
            # Create latent representation
            latent_df = make_latent_series(self.encoder, scaled_features)
            
            # Add RET for sequence creation (needed by to_sequences function)
            latent_df['RET'] = raw_features.loc[latent_df.index, 'RET']
            
            # Take the last window for prediction
            if len(latent_df) < window:
                raise ValueError(f"Insufficient latent data: {len(latent_df)} < {window}")
            
            # Create sequence for prediction (we only need the features, not labels)
            latent_cols = [col for col in latent_df.columns if col != 'RET']
            latent_features = latent_df[latent_cols]
            
            if len(latent_features) < window:
                raise ValueError(f"Need at least {window} days for prediction, got {len(latent_features)}")
            
            # Take the last window days for prediction
            last_sequence = latent_features.iloc[-window:].values
            last_sequence = last_sequence.reshape(1, window, -1)  # Shape for LSTM: (1, window, features)
            
            # Make prediction
            prob = self.lstm.predict(last_sequence, verbose=0)[0, 0]
            
            # Generate signal
            signal = 1 if prob >= 0.5 else 0
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            
            return {
                'probability': prob,
                'signal': signal,
                'confidence': confidence,
                'date': latent_df.index[-1]
            }
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            raise
    
    def run_daily_prediction(self):
        """Run the complete daily prediction pipeline."""
        try:
            print(f"Running live prediction for {date.today()}")
            
            # Fetch market data
            sp_data, vix_data = self.fetch_market_data()
            
            # Prepare features
            raw_features, scaled_features = self.prepare_features(sp_data, vix_data)
            
            # Generate signal
            signal_info = self.generate_signal(raw_features, scaled_features)
            
            # Output results
            print(f"\n{'='*50}")
            print(f"TRADING SIGNAL FOR {date.today()}")
            print(f"{'='*50}")
            print(f"Probability: {signal_info['probability']:.3f}")
            print(f"Signal: {'LONG' if signal_info['signal'] == 1 else 'CASH'}")
            print(f"Confidence: {signal_info['confidence']:.3f}")
            print(f"Based on data through: {signal_info['date']}")
            print(f"{'='*50}")
            
            return signal_info
            
        except Exception as e:
            print(f"Error in daily prediction: {e}")
            return None

def main():
    """Main function for command-line usage."""
    trader = LiveTrader()
    signal_info = trader.run_daily_prediction()
    
    if signal_info is None:
        print("Failed to generate trading signal")
        sys.exit(1)
    
    # Optional: Save signal to file for record keeping
    try:
        log_entry = {
            'date': date.today().isoformat(),
            'probability': signal_info['probability'],
            'signal': signal_info['signal'],
            'confidence': signal_info['confidence']
        }
        
        # Append to CSV log
        log_df = pd.DataFrame([log_entry])
        log_file = 'trading_signals.csv'
        
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, mode='w', header=True, index=False)
        
        print(f"Signal logged to {log_file}")
        
    except Exception as e:
        print(f"Warning: Could not log signal to file: {e}")

if __name__ == "__main__":
    main() 