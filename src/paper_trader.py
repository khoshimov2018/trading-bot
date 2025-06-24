#!/usr/bin/env python3


import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import os
from src.live import LiveTrader
import warnings
warnings.filterwarnings('ignore')

class PaperTrader:
    def __init__(self, initial_capital=100000, position_size=0.1, log_file='paper_trading_log.csv'):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.log_file = log_file
        
        # Portfolio state
        self.current_position = 0  # 0 = cash, 1 = long
        self.entry_price = None
        self.entry_date = None
        
        # Performance tracking
        self.trades = []
        self.daily_returns = []
        self.signals_log = []
        
        # Initialize live trader
        self.live_trader = LiveTrader()
        
        print(f"🎮 Paper Trader initialized")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Position Size: {position_size:.1%}")
        print(f"   Log File: {log_file}")
    
    def get_current_price(self):
        try:
            import yfinance as yf
            ticker = yf.Ticker("^GSPC")
            data = ticker.history(period="1d", interval="1d")
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to recent data
                from src.loader import load_sp500
                sp_data = load_sp500('data/HistoricalData_1750587928127.csv')
                return float(sp_data['Close'].iloc[-1])
        except:
            return 6000.0  # Fallback price
    
    def generate_signal(self):
        try:
            # Get signal from live trader
            sp_data, vix_data = self.live_trader.fetch_market_data(lookback_days=100)
            raw_features, scaled_features = self.live_trader.prepare_features(sp_data, vix_data)
            signal_info = self.live_trader.generate_signal(raw_features, scaled_features)
            
            return signal_info
            
        except Exception as e:
            print(f"⚠️  Error generating signal: {e}")
            return {
                'probability': 0.5,
                'signal': 0,
                'confidence': 0.0,
                'date': datetime.now()
            }
    
    def execute_trade(self, signal_info, current_price):
        """Execute trade based on signal."""
        trade_date = datetime.now()
        
        # Determine action based on signal and current position
        if signal_info['signal'] == 1 and self.current_position == 0:
            # Enter long position
            position_value = self.current_capital * self.position_size
            shares = position_value / current_price
            
            # Transaction costs (2 bps)
            transaction_cost = position_value * 0.0002
            self.current_capital -= transaction_cost
            
            self.current_position = 1
            self.entry_price = current_price
            self.entry_date = trade_date
            
            trade = {
                'date': trade_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': position_value,
                'transaction_cost': transaction_cost,
                'probability': signal_info['probability'],
                'confidence': signal_info['confidence'],
                'capital_after': self.current_capital
            }
            
            self.trades.append(trade)
            print(f"📈 LONG: Bought ${position_value:,.2f} at ${current_price:.2f} (Prob: {signal_info['probability']:.3f})")
            
        elif signal_info['signal'] == 0 and self.current_position == 1:
            # Exit long position
            position_value = self.current_capital * self.position_size
            shares = position_value / self.entry_price  # Original shares
            exit_value = shares * current_price
            
            # Calculate P&L
            pnl = exit_value - position_value
            pnl_pct = pnl / position_value
            
            # Transaction costs
            transaction_cost = exit_value * 0.0002
            net_pnl = pnl - transaction_cost
            
            # Update capital
            self.current_capital += net_pnl
            
            self.current_position = 0
            
            trade = {
                'date': trade_date,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': exit_value,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'transaction_cost': transaction_cost,
                'probability': signal_info['probability'],
                'confidence': signal_info['confidence'],
                'capital_after': self.current_capital,
                'holding_period': (trade_date - self.entry_date).days
            }
            
            self.trades.append(trade)
            print(f"💰 SELL: Sold at ${current_price:.2f}, P&L: ${net_pnl:,.2f} ({pnl_pct:.1%})")
        
        # Log signal regardless of action
        signal_log = {
            'date': trade_date,
            'probability': signal_info['probability'],
            'signal': signal_info['signal'],
            'confidence': signal_info['confidence'],
            'current_price': current_price,
            'position': self.current_position,
            'capital': self.current_capital
        }
        self.signals_log.append(signal_log)
    
    def calculate_performance(self):
        """Calculate current performance metrics."""
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0
            }
        
        # Total return
        total_return = self.current_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Trade statistics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        num_trades = len(completed_trades)
        
        if num_trades > 0:
            winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
            win_rate = winning_trades / num_trades
            avg_return = np.mean([t['pnl'] for t in completed_trades])
        else:
            win_rate = 0.0
            avg_return = 0.0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'current_capital': self.current_capital
        }
    
    def save_logs(self):
        """Save trading logs to files."""
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv('paper_trades.csv', index=False)
            print(f"💾 Saved {len(self.trades)} trades to paper_trades.csv")
        
        # Save signals
        if self.signals_log:
            signals_df = pd.DataFrame(self.signals_log)
            signals_df.to_csv('paper_signals.csv', index=False)
            print(f"💾 Saved {len(self.signals_log)} signals to paper_signals.csv")
    
    def run_paper_trading(self, days=30):
        """
        Run paper trading simulation.
        
        Args:
            days: Number of days to simulate (for demo purposes)
        """
        print(f"\n🎮 Starting Paper Trading Simulation")
        print(f"   Duration: {days} days")
        print("=" * 50)
        
        for day in range(days):
            try:
                # Simulate daily trading
                current_date = date.today() - timedelta(days=days-day-1)
                current_price = self.get_current_price() + np.random.normal(0, 50)  # Add some noise
                
                # Generate signal
                signal_info = self.generate_signal()
                
                # Execute trade
                self.execute_trade(signal_info, current_price)
                
                # Print daily status
                if day % 5 == 0 or day == days - 1:
                    perf = self.calculate_performance()
                    print(f"Day {day+1}: Capital=${perf['current_capital']:,.2f} "
                          f"({perf['total_return_pct']:+.1%}), Trades={perf['num_trades']}")
                
            except Exception as e:
                print(f"⚠️  Error on day {day+1}: {e}")
                continue
        
        # Final performance report
        print("\n" + "=" * 50)
        print("🏁 PAPER TRADING COMPLETE")
        print("=" * 50)
        
        perf = self.calculate_performance()
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Capital:   ${perf['current_capital']:,.2f}")
        print(f"   Total Return:    ${perf['total_return']:,.2f} ({perf['total_return_pct']:+.1%})")
        print(f"   Number of Trades: {perf['num_trades']}")
        print(f"   Win Rate:        {perf['win_rate']:.1%}")
        print(f"   Avg Return/Trade: ${perf['avg_return_per_trade']:,.2f}")
        
        # Save logs
        self.save_logs()
        
        return perf
    
    def monitor_live_performance(self, duration_days=30):
        """
        Monitor live paper trading performance.
        
        This would be called daily in a real implementation.
        """
        print(f"📈 Live Paper Trading Monitor")
        print(f"   Started: {datetime.now()}")
        print(f"   Duration: {duration_days} days")
        print(f"   Check this script daily for updates!")
        
        # In a real implementation, this would:
        # 1. Generate daily signals
        # 2. Track performance vs. benchmarks
        # 3. Monitor for model decay
        # 4. Alert on significant drawdowns
        # 5. Generate daily reports

def create_monitoring_schedule():
    """Create a template for daily monitoring."""
    monitoring_template = """
# Daily Paper Trading Monitoring Checklist

## Pre-Market (Before 9:30 AM ET)
- [ ] Check overnight news and market events
- [ ] Verify data feeds are working
- [ ] Review yesterday's signal accuracy
- [ ] Check for any system errors

## Market Open (9:30 AM ET)
- [ ] Generate today's signal
- [ ] Execute paper trades based on signal
- [ ] Log signal and trade details
- [ ] Monitor initial market reaction

## End of Day (After 4:00 PM ET)
- [ ] Calculate daily P&L
- [ ] Update performance metrics
- [ ] Compare actual vs. predicted performance
- [ ] Check for any anomalies or outliers

## Weekly Review (Fridays)
- [ ] Calculate weekly Sharpe ratio
- [ ] Compare to walk-forward test expectations
- [ ] Monitor hit rate vs. backtest
- [ ] Check for signs of model decay
- [ ] Update risk management parameters if needed

## Monthly Review
- [ ] Full performance analysis vs. robustness test results
- [ ] Correlation analysis with market regimes
- [ ] Model recalibration if necessary
- [ ] Decision on continuing/stopping strategy
"""
    
    with open('paper_trading_monitoring.md', 'w') as f:
        f.write(monitoring_template)
    
    print("📋 Created paper_trading_monitoring.md with daily checklist")

if __name__ == "__main__":
    # Demo paper trading
    trader = PaperTrader(initial_capital=100000, position_size=0.1)
    trader.run_paper_trading(days=30)
    create_monitoring_schedule() 