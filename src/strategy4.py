"""
Strategy 4: Momentum + Risk Parity Hybrid (Momentum Selection with Volatility Weighting)

This strategy combines the best of both worlds: directional momentum selection and 
volatility-balanced weighting. It first picks the assets with positive momentum 
and then allocates among them using an inverse-volatility scheme.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class MomentumRiskParityStrategy:
    """
    Momentum + Risk Parity Hybrid Strategy.
    Step 1: Select assets with above-average 6-month momentum
    Step 2: Apply inverse-volatility weighting to selected assets
    """
    def __init__(self, asset_universe: List[str], momentum_period: int = 126, volatility_window: int = 60):
        self.asset_universe = asset_universe
        self.momentum_period = momentum_period  # 6 months = 126 trading days
        self.volatility_window = volatility_window  # 60 days for volatility

    def calculate_target_weights(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        if len(historical_data) < max(self.momentum_period, self.volatility_window):
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Step 1: Calculate 6-month momentum and select assets
        momentum_data = historical_data[self.asset_universe].iloc[-self.momentum_period:]
        if len(momentum_data) < 2:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Calculate momentum returns: R_6m(i) = (P_i / P_i,-6m) - 1
        start_prices = momentum_data.iloc[0]
        end_prices = momentum_data.iloc[-1]
        momentum_returns = (end_prices / start_prices) - 1
        momentum_returns = momentum_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(momentum_returns) == 0:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Select assets with above-average momentum
        avg_return = momentum_returns.mean()
        selected_assets = momentum_returns[momentum_returns > avg_return].index.tolist()
        
        if len(selected_assets) == 0:
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Step 2: Apply inverse-volatility weighting to selected assets
        vol_data = historical_data[selected_assets].iloc[-self.volatility_window:]
        vols = vol_data.pct_change().std()
        vols = vols.replace(0, np.nan)
        inv_vols = 1 / vols
        inv_vols = inv_vols.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        weights = {}
        if inv_vols.sum() > 0:
            # Normalize inverse volatilities for selected assets
            normalized_inv_vols = inv_vols / inv_vols.sum()
            for asset in self.asset_universe:
                weights[asset] = float(normalized_inv_vols.get(asset, 0.0))
        else:
            # If volatility calculation fails, equal-weight selected assets
            equal_weight = 1.0 / len(selected_assets)
            for asset in self.asset_universe:
                weights[asset] = equal_weight if asset in selected_assets else 0.0
        
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
            'name': 'Momentum + Risk Parity Hybrid',
            'description': 'Momentum selection with inverse-volatility weighting',
            'momentum_period': f"{self.momentum_period} days",
            'volatility_window': f"{self.volatility_window} days",
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