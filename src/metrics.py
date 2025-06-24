import numpy as np
import pandas as pd

def sharpe(returns, risk_free_rate=0.0):

    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  

def max_drawdown(cumulative_returns):

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def information_coefficient(predictions, actual_returns):

    return np.corrcoef(predictions, actual_returns)[0, 1] 