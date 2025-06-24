#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results():
    """Load all results from database"""
    conn = sqlite3.connect('results.db')
    df = pd.read_sql('SELECT * FROM experiments', conn)
    conn.close()
    
    # Convert date columns
    df['test_start'] = pd.to_datetime(df['test_start'])
    df['test_end'] = pd.to_datetime(df['test_end'])
    
    return df

def analyze_edge_robustness(df):
    """Analyze the robustness of the trading edge"""
    print("============================================================")
    print("🔍 EDGE ROBUSTNESS ANALYSIS")
    print("============================================================")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique configurations: {df[['latent_dim', 'window', 'dropout', 'l2_reg']].drop_duplicates().shape[0]}")
    print(f"  Test periods analyzed: {df[['test_start', 'test_end']].drop_duplicates().shape[0]}")
    
    # Overall performance statistics
    print(f"\n📈 PERFORMANCE DISTRIBUTION:")
    print(f"  Mean Sharpe: {df['sharpe'].mean():.3f}")
    print(f"  Median Sharpe: {df['sharpe'].median():.3f}")
    print(f"  Std Sharpe: {df['sharpe'].std():.3f}")
    print(f"  Profitable experiments: {(df['sharpe'] > 0).mean():.1%}")
    print(f"  Sharpe > 1.0: {(df['sharpe'] > 1.0).mean():.1%}")
    print(f"  Sharpe > 2.0: {(df['sharpe'] > 2.0).mean():.1%}")
    
    print(f"\n🎯 HIT RATE ANALYSIS:")
    print(f"  Mean Hit Rate: {df['hit_rate'].mean():.3f}")
    print(f"  Median Hit Rate: {df['hit_rate'].median():.3f}")
    print(f"  Hit Rate > 55%: {(df['hit_rate'] > 0.55).mean():.1%}")
    print(f"  Hit Rate > 60%: {(df['hit_rate'] > 0.60).mean():.1%}")
    
    print(f"\n💰 RISK ANALYSIS:")
    print(f"  Mean Max Drawdown: {df['max_drawdown'].mean():.3f}")
    print(f"  Worst Drawdown: {df['max_drawdown'].min():.3f}")
    print(f"  Drawdown < 10%: {(df['max_drawdown'] > -0.10).mean():.1%}")
    
    # Best performing configurations
    print(f"\n🏆 TOP CONFIGURATIONS BY SHARPE:")
    top_configs = df.nlargest(10, 'sharpe')[['latent_dim', 'window', 'dropout', 'l2_reg', 'sharpe', 'hit_rate', 'max_drawdown']]
    for i, row in top_configs.iterrows():
        print(f"  {int(row['latent_dim']):2d}|{int(row['window']):2d}|{row['dropout']:.1f}|{row['l2_reg']:.1e} → Sharpe: {row['sharpe']:6.2f}, Hit: {row['hit_rate']:.3f}, DD: {row['max_drawdown']:6.1%}")

def analyze_parameter_stability(df):
    """Analyze stability across different hyperparameters"""
    print("\n============================================================")
    print("🔧 HYPERPARAMETER STABILITY ANALYSIS")
    print("============================================================")
    
    # Group by hyperparameters and analyze
    grouped = df.groupby(['latent_dim', 'window', 'dropout', 'l2_reg'])
    
    config_stats = grouped.agg({
        'sharpe': ['mean', 'std', 'min', 'max'],
        'hit_rate': ['mean', 'std'],
        'max_drawdown': ['mean', 'min']
    }).round(3)
    
    config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns]
    config_stats = config_stats.reset_index()
    
    print(f"\n📊 CONFIGURATION PERFORMANCE SUMMARY:")
    print(f"{'Config':<20} {'Mean Sharpe':<12} {'Sharpe Std':<12} {'Min Sharpe':<12} {'Max Sharpe':<12} {'Hit Rate':<10}")
    print("-" * 80)
    
    # Sort by mean Sharpe
    config_stats_sorted = config_stats.sort_values('sharpe_mean', ascending=False)
    
    for _, row in config_stats_sorted.head(15).iterrows():
        config_str = f"{int(row['latent_dim']):2d}|{int(row['window']):2d}|{row['dropout']:.1f}|{row['l2_reg']:.1e}"
        print(f"{config_str:<20} {row['sharpe_mean']:11.3f} {row['sharpe_std']:11.3f} {row['sharpe_min']:11.3f} {row['sharpe_max']:11.3f} {row['hit_rate_mean']:9.3f}")
    
    # Find most consistent performers
    print(f"\n🎯 MOST CONSISTENT CONFIGURATIONS (Low Std):")
    consistent = config_stats_sorted.nsmallest(10, 'sharpe_std')
    for _, row in consistent.iterrows():
        if row['sharpe_mean'] > 0:  # Only positive mean Sharpe
            config_str = f"{int(row['latent_dim']):2d}|{int(row['window']):2d}|{row['dropout']:.1f}|{row['l2_reg']:.1e}"
            print(f"  {config_str} → Mean: {row['sharpe_mean']:6.3f}, Std: {row['sharpe_std']:6.3f}")

def analyze_time_stability(df):
    """Analyze performance across different time periods"""
    print("\n============================================================")
    print("📅 TIME PERIOD STABILITY ANALYSIS")
    print("============================================================")
    
    # Extract year from test period
    df['test_year'] = df['test_start'].dt.year
    
    # Performance by year
    yearly_stats = df.groupby('test_year').agg({
        'sharpe': ['mean', 'std', 'count'],
        'hit_rate': 'mean',
        'max_drawdown': 'mean'
    }).round(3)
    
    yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]
    
    print(f"\n📊 PERFORMANCE BY TEST YEAR:")
    print(f"{'Year':<6} {'Mean Sharpe':<12} {'Sharpe Std':<12} {'Experiments':<12} {'Hit Rate':<10} {'Avg DD':<10}")
    print("-" * 70)
    
    for year, row in yearly_stats.iterrows():
        print(f"{year:<6} {row['sharpe_mean']:11.3f} {row['sharpe_std']:11.3f} {row['sharpe_count']:11.0f} {row['hit_rate_mean']:9.3f} {row['max_drawdown_mean']:9.3f}")
    
    # Market regime analysis
    print(f"\n📈 MARKET REGIME ANALYSIS:")
    crisis_years = [2020, 2022]  # COVID, Ukraine war
    crisis_mask = df['test_year'].isin(crisis_years)
    
    print(f"  Crisis periods (2020, 2022):")
    print(f"    Mean Sharpe: {df[crisis_mask]['sharpe'].mean():.3f}")
    print(f"    Hit Rate: {df[crisis_mask]['hit_rate'].mean():.3f}")
    print(f"    Worst DD: {df[crisis_mask]['max_drawdown'].min():.3f}")
    
    print(f"  Normal periods:")
    print(f"    Mean Sharpe: {df[~crisis_mask]['sharpe'].mean():.3f}")
    print(f"    Hit Rate: {df[~crisis_mask]['hit_rate'].mean():.3f}")
    print(f"    Worst DD: {df[~crisis_mask]['max_drawdown'].min():.3f}")

def generate_recommendations(df):
    """Generate actionable recommendations"""
    print("\n============================================================")
    print("💡 RECOMMENDATIONS")
    print("============================================================")
    
    # Find best overall configurations
    grouped = df.groupby(['latent_dim', 'window', 'dropout', 'l2_reg'])
    config_stats = grouped.agg({
        'sharpe': ['mean', 'std'],
        'hit_rate': 'mean',
        'max_drawdown': 'mean'
    })
    
    config_stats.columns = ['_'.join(col).strip() for col in config_stats.columns]
    config_stats = config_stats.reset_index()
    
    # Score configs based on multiple criteria
    config_stats['score'] = (
        config_stats['sharpe_mean'] * 0.4 +  # 40% weight on returns
        (config_stats['hit_rate_mean'] - 0.5) * 10 * 0.3 +  # 30% weight on hit rate above 50%
        -config_stats['sharpe_std'] * 0.2 +  # 20% weight on consistency
        config_stats['max_drawdown_mean'] * 0.1  # 10% weight on drawdown (negative, so better DD = higher score)
    )
    
    best_configs = config_stats.nlargest(5, 'score')
    
    print(f"\n🏆 TOP 5 RECOMMENDED CONFIGURATIONS:")
    print(f"{'Rank':<5} {'Config':<20} {'Mean Sharpe':<12} {'Hit Rate':<10} {'Consistency':<12} {'Score':<8}")
    print("-" * 75)
    
    for idx, (_, row) in enumerate(best_configs.iterrows(), 1):
        config_str = f"{int(row['latent_dim']):2d}|{int(row['window']):2d}|{row['dropout']:.1f}|{row['l2_reg']:.1e}"
        print(f"{idx:<5} {config_str:<20} {row['sharpe_mean']:11.3f} {row['hit_rate_mean']:9.3f} {row['sharpe_std']:11.3f} {row['score']:7.3f}")
    
    # Overall assessment
    overall_profitable = (df['sharpe'] > 0).mean()
    overall_mean_sharpe = df['sharpe'].mean()
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if overall_mean_sharpe > 0.5 and overall_profitable > 0.6:
        print("  ✅ STRONG EDGE: Strategy shows consistent positive returns")
    elif overall_mean_sharpe > 0.2 and overall_profitable > 0.5:
        print("  ⚠️  MODERATE EDGE: Strategy has potential but needs refinement")
    else:
        print("  ❌ WEAK/NO EDGE: Strategy does not demonstrate reliable alpha")
    
    print(f"\n📋 NEXT STEPS:")
    if overall_mean_sharpe > 0.3:
        print("  1. Implement the top configuration in paper trading")
        print("  2. Monitor performance for 3-6 months")
        print("  3. Consider ensemble of top 3 configurations")
        print("  4. Implement dynamic position sizing")
    else:
        print("  1. Strategy needs significant improvement before live trading")
        print("  2. Consider alternative features or model architectures")
        print("  3. Investigate data leakage or overfitting issues")
        print("  4. Back to research phase - not ready for deployment")

def create_visualizations(df):
    """Create key visualizations"""
    print("\n============================================================")
    print("📊 GENERATING VISUALIZATIONS")
    print("============================================================")
    
    # Create output directory
    Path('robustness_plots').mkdir(exist_ok=True)
    
    # 1. Sharpe distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['sharpe'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(df['sharpe'].mean(), color='red', linestyle='--', label=f'Mean: {df["sharpe"].mean():.3f}')
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sharpe Ratios')
    plt.legend()
    
    # 2. Hit rate distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['hit_rate'], bins=30, alpha=0.7, edgecolor='black', color='green')
    plt.axvline(df['hit_rate'].mean(), color='red', linestyle='--', label=f'Mean: {df["hit_rate"].mean():.3f}')
    plt.axvline(0.5, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Hit Rate')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hit Rates')
    plt.legend()
    
    # 3. Sharpe vs Hit Rate scatter
    plt.subplot(2, 2, 3)
    plt.scatter(df['hit_rate'], df['sharpe'], alpha=0.5)
    plt.xlabel('Hit Rate')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe vs Hit Rate')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(0.5, color='black', linestyle='-', alpha=0.3)
    
    # 4. Performance by year
    plt.subplot(2, 2, 4)
    df['test_year'] = df['test_start'].dt.year
    yearly_sharpe = df.groupby('test_year')['sharpe'].mean()
    plt.bar(yearly_sharpe.index, yearly_sharpe.values, alpha=0.7)
    plt.xlabel('Test Year')
    plt.ylabel('Mean Sharpe Ratio')
    plt.title('Performance by Test Year')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robustness_plots/summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Saved summary_analysis.png")
    
    # Configuration heatmap
    pivot_data = df.groupby(['latent_dim', 'window'])['sharpe'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
    plt.title('Mean Sharpe Ratio by Configuration')
    plt.ylabel('Latent Dimension')
    plt.xlabel('Window Size')
    plt.savefig('robustness_plots/config_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Saved config_heatmap.png")

def main():
    print("🔍 AE-LSTM Trading Strategy - Robustness Test Analysis")
    print("============================================================")
    
    # Load data
    df = load_results()
    print(f"✅ Loaded {len(df)} experiment results")
    
    # Run analysis
    analyze_edge_robustness(df)
    analyze_parameter_stability(df)
    analyze_time_stability(df)
    generate_recommendations(df)
    create_visualizations(df)
    
    print(f"\n🎉 Analysis complete! Check robustness_plots/ for visualizations.")

if __name__ == "__main__":
    main() 