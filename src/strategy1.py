"""
Strategy 1: Equal-Weight Portfolio (Monthly Rebalanced)

This strategy allocates equal capital weight to each asset class in the universe.
At inception and after each rebalance, each asset gets weight w_i = 1/N (for N assets).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EqualWeightStrategy:
    """
    Equal-Weight Portfolio Strategy with monthly rebalancing.
    
    This is a naïve baseline strategy that allocates equal capital weight to each 
    asset class in the universe. At inception (and after each rebalance), each 
    asset gets weight w_i = 1/N (for N assets).
    """
    
    def __init__(self, asset_universe: List[str], rebalancing_frequency: str = 'monthly'):
        """
        Initialize the Equal-Weight Strategy.
        
        Args:
            asset_universe: List of asset symbols
            rebalancing_frequency: Rebalancing frequency ('monthly', 'quarterly', etc.)
        """
        self.asset_universe = asset_universe
        self.rebalancing_frequency = rebalancing_frequency
        self.n_assets = len(asset_universe)
        self.equal_weight = 1.0 / self.n_assets
        
        logger.info(f"Initialized Equal-Weight Strategy with {self.n_assets} assets")
        logger.info(f"Equal weight per asset: {self.equal_weight:.2%}")
    
    def calculate_target_weights(self, current_date: datetime, 
                               historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate target weights for equal-weight allocation.
        
        Args:
            current_date: Current date for weight calculation
            historical_data: Historical price data for all assets
            
        Returns:
            Dictionary mapping asset symbols to target weights
        """
        if len(historical_data) < 252:
            return {asset: 0.0 for asset in self.asset_universe}
        # Equal weight allocation - each asset gets 1/N weight
        target_weights = {asset: self.equal_weight for asset in self.asset_universe}
        
        logger.info(f"Equal-weight allocation calculated for {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"Target weights: {target_weights}")
        
        return target_weights
    
    def get_daily_signals(self, current_date: datetime, 
                         historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Get daily portfolio weights (signals) for the strategy.
        
        Args:
            current_date: Current date
            historical_data: Historical price data for all assets
            
        Returns:
            Dictionary mapping asset symbols to current portfolio weights
        """
        if len(historical_data) < 252:
            return {asset: 0.0 for asset in self.asset_universe}
        return self.calculate_target_weights(current_date, historical_data)
    
    def should_rebalance(self, current_date: datetime, 
                        last_rebalance_date: datetime) -> bool:
        """
        Determine if rebalancing is needed based on frequency.
        
        Args:
            current_date: Current date
            last_rebalance_date: Date of last rebalancing
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        if self.rebalancing_frequency == 'monthly':
            # Rebalance if we're in a different month
            return (current_date.year != last_rebalance_date.year or 
                   current_date.month != last_rebalance_date.month)
        elif self.rebalancing_frequency == 'quarterly':
            # Rebalance if we're in a different quarter
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance_date.month - 1) // 3
            return (current_date.year != last_rebalance_date.year or 
                   current_quarter != last_quarter)
        else:
            # Default to monthly
            return (current_date.year != last_rebalance_date.year or 
                   current_date.month != last_rebalance_date.month)
    
    def calculate_weight_drift(self, current_weights: Dict[str, float], 
                             target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weight drift from target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Dictionary of weight drift for each asset
        """
        drift = {}
        for asset in self.asset_universe:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            drift[asset] = current - target
        
        return drift
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get strategy information and description.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'Equal-Weight Portfolio',
            'description': 'Equal capital weight allocation to all assets',
            'rebalancing_frequency': self.rebalancing_frequency,
            'n_assets': self.n_assets,
            'equal_weight': f"{self.equal_weight:.2%}"
        }
    
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