"""
Strategy 2: Volatility-Adjusted Risk Parity (60-Day Inverse-Vol Weighting)

This strategy allocates weights inversely proportional to each asset's volatility,
aiming for an equal risk contribution from each asset. It's a risk parity approach
that gives larger weights to historically lower-volatility assets (like bonds) and
smaller weights to higher-volatility assets (like equities or silver).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class VolatilityAdjustedStrategy:
    """
    Volatility-Adjusted Risk Parity Strategy (Inverse-Volatility Weighting).
    """
    def __init__(self, asset_universe: List[str], volatility_window: int = 60):
        self.asset_universe = asset_universe
        self.volatility_window = volatility_window

    def calculate_target_weights(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        if len(historical_data) < self.volatility_window:
            return {asset: 0.0 for asset in self.asset_universe}
        # Compute rolling window
        window_data = historical_data[self.asset_universe].iloc[-self.volatility_window:]
        vols = window_data.pct_change().std()
        # Avoid division by zero
        vols = vols.replace(0, np.nan)
        inv_vols = 1 / vols
        inv_vols = inv_vols.replace([np.inf, -np.inf], np.nan).fillna(0)
        if inv_vols.sum() == 0:
            weights = {asset: 1.0 / len(self.asset_universe) for asset in self.asset_universe}
        else:
            weights = (inv_vols / inv_vols.sum()).to_dict()
        # Ensure all assets in output
        return {asset: float(weights.get(asset, 0.0)) for asset in self.asset_universe}

    def get_daily_signals(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        return self.calculate_target_weights(current_date, historical_data)

    def calculate_volatilities(self, historical_data: pd.DataFrame, 
                             current_date: datetime) -> Dict[str, float]:
        """
        Calculate rolling volatilities for all assets.
        
        Args:
            historical_data: Historical price data for all assets
            current_date: Current date for volatility calculation
            
        Returns:
            Dictionary mapping asset symbols to their volatilities
        """
        volatilities = {}
        
        # Calculate end date for volatility window
        end_date = current_date
        start_date = end_date - timedelta(days=self.volatility_window)
        
        for asset in self.asset_universe:
            if asset in historical_data.columns:
                # Get price data for the volatility window
                asset_data = historical_data[asset].loc[start_date:end_date]
                
                if len(asset_data) >= self.volatility_window * 0.8:  # At least 80% of data
                    # Calculate daily returns
                    returns = asset_data.pct_change().dropna()
                    
                    if len(returns) > 0:
                        # Calculate volatility (standard deviation of returns)
                        volatility = returns.std()
                        
                        # Apply volatility floor
                        volatility = max(volatility, 0.01)
                        
                        volatilities[asset] = volatility
                    else:
                        volatilities[asset] = 0.01
                else:
                    volatilities[asset] = 0.01
            else:
                volatilities[asset] = 0.01
        
        logger.info(f"Calculated volatilities for {len(volatilities)} assets")
        return volatilities
    
    def calculate_risk_contributions(self, weights: Dict[str, float], 
                                   volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio volatility.
        
        Args:
            weights: Portfolio weights
            volatilities: Asset volatilities
            
        Returns:
            Dictionary of risk contributions for each asset
        """
        risk_contributions = {}
        for asset in self.asset_universe:
            weight = weights.get(asset, 0.0)
            vol = volatilities.get(asset, 0.0)
            risk_contributions[asset] = weight * vol
        
        return risk_contributions
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get strategy information and description.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'Volatility-Adjusted Risk Parity',
            'description': 'Inverse-volatility weighting for equal risk contribution',
            'volatility_window': f"{self.volatility_window} days",
            'volatility_floor': f"{0.01:.2%}",
            'n_assets': len(self.asset_universe)
        }

    def should_rebalance(self, current_date: datetime, last_rebalance_date: datetime) -> bool:
        """Check if rebalancing is needed (monthly)."""
        return (current_date.year != last_rebalance_date.year or 
                current_date.month != last_rebalance_date.month)
    
    def get_weight_history(self, historical_data: pd.DataFrame, min_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Get full weight history and rebalance-only history for the strategy.
        
        Args:
            historical_data: Historical price data
            min_days: Minimum days before starting calculations
            
        Returns:
            Dictionary with 'full_history' and 'rebalance_history' DataFrames
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