#!/usr/bin/env python3
"""
Simplified Conservative Live Test - Auto-run for demonstration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conservative_live_test import ConservativeLiveTest

def main():
    """Run conservative test automatically."""
    print("🚨 AUTOMATED RISK ACKNOWLEDGMENT 🚨")
    print("=" * 60)
    print("This strategy showed NEGATIVE expected returns in testing.")
    print("Mean Sharpe: -0.335 | Only 47.7% profitable experiments")
    print("Running MINIMAL RISK demonstration with $50 max positions")
    print("=" * 60)
    
    # Run the test automatically
    test = ConservativeLiveTest()
    test.run_test_cycle(days=3)  # Shorter test
    
    print("\n" + "="*60)
    print("DEMO COMPLETE - Remember this strategy has negative expected returns!")
    print("="*60)

if __name__ == "__main__":
    main() 