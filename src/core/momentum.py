"""
ReSolve Momentum Calculation Implementation
Implements the core momentum methodology from the ReSolve paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from ..core.base import IMomentumCalculator, ValidationError


class ResolveTimeSerieMomentumCalculator(IMomentumCalculator):
    """
    Implements ReSolve's time-series momentum methodology.
    
    The ReSolve approach uses multiple lookback periods to calculate momentum scores,
    with equal weighting across different time horizons to capture both short-term
    and long-term momentum effects.
    """
    
    def __init__(self, 
                 equal_weight_periods: bool = True,
                 minimum_observations: int = 21,
                 volatility_adjustment: bool = True):
        """
        Initialize momentum calculator.
        
        Args:
            equal_weight_periods: Whether to equally weight all lookback periods
            minimum_observations: Minimum number of observations required
            volatility_adjustment: Whether to adjust momentum by volatility
        """
        self.equal_weight_periods = equal_weight_periods
        self.minimum_observations = minimum_observations
        self.volatility_adjustment = volatility_adjustment
        self.logger = logging.getLogger(__name__)
        
    def calculate_momentum_score(self, price_data: pd.DataFrame, lookback_periods: List[int]) -> float:
        """
        Calculate momentum score for single asset using ReSolve methodology.
        
        The ReSolve momentum score is calculated as:
        1. Calculate returns for each lookback period
        2. Optionally adjust by volatility (risk-adjusted momentum)
        3. Combine using equal or custom weighting
        
        Args:
            price_data: DataFrame with price data (must have 'close' column)
            lookback_periods: List of lookback periods in business days
            
        Returns:
            Momentum score (higher = stronger momentum)
        """
        if 'close' not in price_data.columns:
            raise ValidationError("Price data must contain 'close' column")
            
        if len(price_data) < max(lookback_periods) + self.minimum_observations:
            raise ValidationError(f"Insufficient data: need at least {max(lookback_periods) + self.minimum_observations} observations")
        
        prices = price_data['close'].dropna()
        momentum_components = []
        
        for period in lookback_periods:
            if len(prices) < period + 1:
                self.logger.warning(f"Insufficient data for {period}-day lookback")
                continue
                
            # Calculate total return over lookback period
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-(period + 1)]
            total_return = (current_price / past_price) - 1
            
            if self.volatility_adjustment:
                # Calculate volatility over the same period for risk adjustment
                returns = prices.pct_change().dropna()
                if len(returns) >= period:
                    period_returns = returns.iloc[-period:]
                    volatility = period_returns.std() * np.sqrt(252)  # Annualized
                    
                    # Risk-adjusted momentum (avoid division by zero)
                    if volatility > 0.001:  # Minimum volatility threshold
                        risk_adjusted_momentum = total_return / volatility
                    else:
                        risk_adjusted_momentum = total_return
                    
                    momentum_components.append(risk_adjusted_momentum)
                else:
                    momentum_components.append(total_return)
            else:
                momentum_components.append(total_return)
        
        if not momentum_components:
            raise ValidationError("No valid momentum components calculated")
        
        # Combine momentum components
        if self.equal_weight_periods:
            momentum_score = np.mean(momentum_components)
        else:
            # Weight longer periods more heavily (ReSolve approach)
            weights = np.array(lookback_periods[:len(momentum_components)])
            weights = weights / weights.sum()
            momentum_score = np.average(momentum_components, weights=weights)
        
        self.logger.debug(f"Calculated momentum score: {momentum_score:.4f}")
        return momentum_score
    
    def calculate_momentum_signals(self, price_data: Dict[str, pd.DataFrame], 
                                 lookback_periods: List[int]) -> Dict[str, float]:
        """
        Calculate momentum signals for multiple assets.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            lookback_periods: List of lookback periods in business days
            
        Returns:
            Dictionary mapping symbols to momentum scores
        """
        momentum_signals = {}
        
        for symbol, data in price_data.items():
            try:
                momentum_score = self.calculate_momentum_score(data, lookback_periods)
                momentum_signals[symbol] = momentum_score
                self.logger.debug(f"Momentum signal for {symbol}: {momentum_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to calculate momentum for {symbol}: {str(e)}")
                momentum_signals[symbol] = 0.0  # Default to neutral momentum
        
        return momentum_signals
    
    def calculate_momentum_rankings(self, momentum_signals: Dict[str, float]) -> Dict[str, int]:
        """
        Rank assets by momentum score (1 = highest momentum).
        
        Args:
            momentum_signals: Dictionary of momentum scores
            
        Returns:
            Dictionary mapping symbols to rankings
        """
        sorted_symbols = sorted(momentum_signals.items(), key=lambda x: x[1], reverse=True)
        rankings = {symbol: rank + 1 for rank, (symbol, _) in enumerate(sorted_symbols)}
        
        self.logger.info(f"Momentum rankings calculated for {len(rankings)} assets")
        return rankings
    
    def get_momentum_percentiles(self, momentum_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Convert momentum scores to percentiles (0-100).
        
        Args:
            momentum_signals: Dictionary of momentum scores
            
        Returns:
            Dictionary mapping symbols to percentile scores
        """
        scores = list(momentum_signals.values())
        percentiles = {}
        
        for symbol, score in momentum_signals.items():
            percentile = (np.sum(np.array(scores) <= score) / len(scores)) * 100
            percentiles[symbol] = percentile
        
        return percentiles


class CrossSectionalMomentumCalculator(IMomentumCalculator):
    """
    Implements cross-sectional momentum (relative momentum across assets).
    
    This approach ranks assets relative to each other rather than using
    absolute momentum scores.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = None,
                 min_assets_for_ranking: int = 3):
        """
        Initialize cross-sectional momentum calculator.
        
        Args:
            lookback_periods: Default lookback periods if not provided
            min_assets_for_ranking: Minimum assets needed for cross-sectional ranking
        """
        self.default_lookback_periods = lookback_periods or [21, 63, 126, 252]
        self.min_assets_for_ranking = min_assets_for_ranking
        self.logger = logging.getLogger(__name__)
    
    def calculate_momentum_score(self, price_data: pd.DataFrame, lookback_periods: List[int]) -> float:
        """
        Calculate momentum score for single asset.
        Note: Cross-sectional momentum requires multiple assets, so this returns simple momentum.
        """
        if 'close' not in price_data.columns:
            raise ValidationError("Price data must contain 'close' column")
        
        prices = price_data['close'].dropna()
        momentum_components = []
        
        for period in lookback_periods:
            if len(prices) >= period + 1:
                total_return = (prices.iloc[-1] / prices.iloc[-(period + 1)]) - 1
                momentum_components.append(total_return)
        
        return np.mean(momentum_components) if momentum_components else 0.0
    
    def calculate_momentum_signals(self, price_data: Dict[str, pd.DataFrame], 
                                 lookback_periods: List[int]) -> Dict[str, float]:
        """
        Calculate cross-sectional momentum signals.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            lookback_periods: List of lookback periods
            
        Returns:
            Dictionary mapping symbols to normalized momentum scores
        """
        if len(price_data) < self.min_assets_for_ranking:
            raise ValidationError(f"Need at least {self.min_assets_for_ranking} assets for cross-sectional momentum")
        
        # Calculate raw momentum scores for all assets
        raw_momentum = {}
        for symbol, data in price_data.items():
            try:
                momentum_score = self.calculate_momentum_score(data, lookback_periods)
                raw_momentum[symbol] = momentum_score
            except Exception as e:
                self.logger.error(f"Failed to calculate momentum for {symbol}: {str(e)}")
                raw_momentum[symbol] = 0.0
        
        # Normalize scores (z-score normalization)
        scores = list(raw_momentum.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        normalized_momentum = {}
        for symbol, score in raw_momentum.items():
            if std_score > 0:
                normalized_score = (score - mean_score) / std_score
            else:
                normalized_score = 0.0
            normalized_momentum[symbol] = normalized_score
        
        self.logger.info(f"Calculated cross-sectional momentum for {len(normalized_momentum)} assets")
        return normalized_momentum


class CombinedMomentumCalculator(IMomentumCalculator):
    """
    Combines time-series and cross-sectional momentum approaches.
    
    This implementation follows the ReSolve methodology of using both
    absolute and relative momentum signals.
    """
    
    def __init__(self, 
                 ts_weight: float = 0.5,
                 cs_weight: float = 0.5,
                 lookback_periods: List[int] = None):
        """
        Initialize combined momentum calculator.
        
        Args:
            ts_weight: Weight for time-series momentum
            cs_weight: Weight for cross-sectional momentum  
            lookback_periods: Lookback periods in business days
        """
        if abs(ts_weight + cs_weight - 1.0) > 1e-6:
            raise ValidationError("Time-series and cross-sectional weights must sum to 1.0")
        
        self.ts_weight = ts_weight
        self.cs_weight = cs_weight
        self.lookback_periods = lookback_periods or [21, 63, 126, 252]  # 1M, 3M, 6M, 12M
        
        self.ts_calculator = ResolveTimeSerieMomentumCalculator()
        self.cs_calculator = CrossSectionalMomentumCalculator()
        self.logger = logging.getLogger(__name__)
    
    def calculate_momentum_score(self, price_data: pd.DataFrame, lookback_periods: List[int]) -> float:
        """
        Calculate combined momentum score for single asset.
        Note: Cross-sectional component requires multiple assets.
        """
        return self.ts_calculator.calculate_momentum_score(price_data, lookback_periods)
    
    def calculate_momentum_signals(self, price_data: Dict[str, pd.DataFrame], 
                                 lookback_periods: List[int] = None) -> Dict[str, float]:
        """
        Calculate combined momentum signals using both approaches.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            lookback_periods: List of lookback periods (uses default if None)
            
        Returns:
            Dictionary mapping symbols to combined momentum scores
        """
        periods = lookback_periods or self.lookback_periods
        
        # Calculate time-series momentum
        ts_signals = self.ts_calculator.calculate_momentum_signals(price_data, periods)
        
        # Calculate cross-sectional momentum
        cs_signals = self.cs_calculator.calculate_momentum_signals(price_data, periods)
        
        # Combine signals
        combined_signals = {}
        for symbol in price_data.keys():
            ts_score = ts_signals.get(symbol, 0.0)
            cs_score = cs_signals.get(symbol, 0.0)
            
            combined_score = (self.ts_weight * ts_score) + (self.cs_weight * cs_score)
            combined_signals[symbol] = combined_score
        
        self.logger.info(f"Calculated combined momentum signals for {len(combined_signals)} assets")
        return combined_signals
    
    def get_momentum_decomposition(self, price_data: Dict[str, pd.DataFrame], 
                                 lookback_periods: List[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Get detailed momentum decomposition for analysis.
        
        Returns:
            Dictionary with time-series, cross-sectional, and combined scores for each asset
        """
        periods = lookback_periods or self.lookback_periods
        
        ts_signals = self.ts_calculator.calculate_momentum_signals(price_data, periods)
        cs_signals = self.cs_calculator.calculate_momentum_signals(price_data, periods)
        combined_signals = self.calculate_momentum_signals(price_data, periods)
        
        decomposition = {}
        for symbol in price_data.keys():
            decomposition[symbol] = {
                'time_series': ts_signals.get(symbol, 0.0),
                'cross_sectional': cs_signals.get(symbol, 0.0),
                'combined': combined_signals.get(symbol, 0.0)
            }
        
        return decomposition

