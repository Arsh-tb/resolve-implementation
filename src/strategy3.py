"""
Strategy 3: Momentum-Based Selection (6-Month Return Momentum Strategy)

This strategy adds a momentum tilt – it dynamically selects assets based on their 
recent performance. The idea is to hold the asset classes that are trending upward 
relative to others, on the premise that recent winners will continue to outperform.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
from datetime import datetime

class MomentumSelectionStrategy:
    """
    Momentum-Based Selection Strategy (6-Month Return Momentum).
    """
    def __init__(self, asset_universe: List[str], momentum_period: int = 126):
        self.asset_universe = asset_universe
        self.momentum_period = momentum_period  # 6 months = 126 trading days

    def calculate_target_weights(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        if len(historical_data) < self.momentum_period:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Calculate 6-month returns for each asset
        window_data = historical_data[self.asset_universe].iloc[-self.momentum_period:]
        if len(window_data) < 2:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # R_6m(i) = (P_i / P_i,-6m) - 1
        start_prices = window_data.iloc[0]
        end_prices = window_data.iloc[-1]
        momentum_returns = (end_prices / start_prices) - 1
        
        # Remove any NaN or infinite values
        momentum_returns = momentum_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(momentum_returns) == 0:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Compute cross-asset average return
        avg_return = momentum_returns.mean()
        
        # Select assets with above-average momentum
        selected_assets = momentum_returns[momentum_returns > avg_return].index.tolist()
        
        # Equal-weight selected assets
        weights = {}
        if len(selected_assets) > 0:
            equal_weight = 1.0 / len(selected_assets)
            for asset in self.asset_universe:
                weights[asset] = equal_weight if asset in selected_assets else 0.0
        else:
            # If no assets selected, all weights are zero
            weights = {asset: 0.0 for asset in self.asset_universe}
        
        return weights

    def get_daily_signals(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        return self.calculate_target_weights(current_date, historical_data)
    
    def should_rebalance(self, current_date: datetime, last_rebalance_date: datetime) -> bool:
        """Check if rebalancing is needed (monthly)."""
        return (current_date.year != last_rebalance_date.year or 
                current_date.month != last_rebalance_date.month)
    
    def get_strategy_info(self) -> Dict[str, str]:
        """Get strategy information and description."""
        return {
            'name': 'Momentum-Based Selection',
            'description': 'Select assets with above-average 6-month momentum',
            'momentum_period': f"{self.momentum_period} days",
            'n_assets': len(self.asset_universe)
        }
    
    def get_weight_history(self, historical_data: pd.DataFrame, min_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Get full weight history and rebalance-only history for the strategy.
        """
        if len(historical_data) < min_days:
            return {'full_history': pd.DataFrame(), 'rebalance_history': pd.DataFrame()}
        
        date_index = historical_data.index[min_days:]
        weights_history = []
        rebalance_weights = []
        rebalance_dates = []
        last_rebalance_date = None
        
        for current_date in date_index:
            hist_slice = historical_data.loc[:current_date]
            weights = self.calculate_target_weights(current_date, hist_slice)
            weights_history.append(weights)
            
            # Check if this is a rebalance date
            if last_rebalance_date is None or self.should_rebalance(current_date, last_rebalance_date):
                rebalance_weights.append(weights)
                rebalance_dates.append(current_date)
                last_rebalance_date = current_date
        
        full_df = pd.DataFrame(weights_history, index=date_index)
        rebalance_df = pd.DataFrame(rebalance_weights, index=rebalance_dates)
        
        return {
            'full_history': full_df,
            'rebalance_history': rebalance_df
        } 