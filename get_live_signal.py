#!/usr/bin/env python3


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.live import LiveTrader
from datetime import datetime

def main():
    print("🔴 LIVE SIGNAL GENERATION")
    print("=" * 50)
    print("⚠️  WARNING: This strategy has negative expected returns")
    print("⚠️  Robustness test: -0.335 mean Sharpe, 47.7% profitable")
    print("⚠️  Use for research/learning purposes only!")
    print("=" * 50)
    
    try:
        # Initialize live trader
        print("📊 Loading trained models...")
        trader = LiveTrader()
        
        # Generate current signal
        print("📡 Fetching live market data...")
        signal_info = trader.run_daily_prediction()
        
        if signal_info:
            print(f"\n✅ SIGNAL GENERATED SUCCESSFULLY")
            print("=" * 40)
            print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📈 Signal: {'🟢 LONG' if signal_info['signal'] == 1 else '🔴 CASH'}")
            print(f"🎯 Probability: {signal_info['probability']:.1%}")
            print(f"🔧 Confidence: {signal_info['confidence']:.1%}")
            print(f"📊 Based on data through: {signal_info['date'].strftime('%Y-%m-%d')}")
            print("=" * 40)
            
            # Interpretation
            if signal_info['signal'] == 1:
                print("📝 INTERPRETATION:")
                print("   • Model suggests going LONG on S&P 500")
                print("   • Expects positive returns in next period")
                print(f"   • Confidence level: {signal_info['confidence']:.1%}")
            else:
                print("📝 INTERPRETATION:")
                print("   • Model suggests staying in CASH")
                print("   • Expects negative/neutral returns")
                print(f"   • Confidence level: {signal_info['confidence']:.1%}")
            
            print("\n⚠️  IMPORTANT DISCLAIMERS:")
            print("   • This signal is for research purposes only")
            print("   • Strategy showed negative expected returns in testing")
            print("   • Do not use real money without extensive additional validation")
            print("   • Past performance does not guarantee future results")
            
        else:
            print("❌ Failed to generate signal")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure models are trained and data is available")

if __name__ == "__main__":
    main() 