"""
ReSolve Adaptive Asset Allocation Strategy Engine
Main orchestrator that combines all components to implement the complete ReSolve methodology.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from ..core.base import (
    IDataProvider, IMomentumCalculator, IRiskCalculator, IPortfolioOptimizer,
    IRebalancingStrategy, StrategyConfiguration, AssetUniverse, BacktestError
)
from ..core.momentum import CombinedMomentumCalculator
from ..risk.calculator import ResolveRiskCalculator
from ..optimization.optimizer import ResolvePortfolioOptimizer


class ResolveStrategyEngine:
    """
    Main strategy engine implementing the complete ReSolve Adaptive Asset Allocation methodology.
    
    This class orchestrates all components:
    1. Data retrieval and validation
    2. Momentum signal calculation
    3. Risk estimation and covariance matrix construction
    4. Portfolio optimization with momentum signals
    5. Volatility targeting and leverage adjustment
    6. Rebalancing logic
    """
    
    def __init__(self,
                 data_provider: IDataProvider,
                 asset_universe: AssetUniverse,
                 config: StrategyConfiguration,
                 momentum_calculator: Optional[IMomentumCalculator] = None,
                 risk_calculator: Optional[IRiskCalculator] = None,
                 portfolio_optimizer: Optional[IPortfolioOptimizer] = None):
        """
        Initialize ReSolve strategy engine.
        
        Args:
            data_provider: Data provider for price and asset information
            asset_universe: Universe of assets to trade
            config: Strategy configuration parameters
            momentum_calculator: Momentum calculation implementation
            risk_calculator: Risk calculation implementation
            portfolio_optimizer: Portfolio optimization implementation
        """
        self.data_provider = data_provider
        self.asset_universe = asset_universe
        self.config = config
        
        # Initialize components with defaults if not provided
        self.momentum_calculator = momentum_calculator or CombinedMomentumCalculator()
        self.risk_calculator = risk_calculator or ResolveRiskCalculator()
        self.portfolio_optimizer = portfolio_optimizer or ResolvePortfolioOptimizer()
        
        # Initialize state
        self.current_weights = {}
        self.last_rebalance_date = None
        self.price_data_cache = {}
        self.returns_data_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("ReSolve Strategy Engine initialized")
        
    def calculate_target_weights(self, current_date: datetime) -> Dict[str, float]:
        """
        Calculate target portfolio weights for given date using ReSolve methodology.
        
        Args:
            current_date: Date for which to calculate weights
            
        Returns:
            Dictionary of target weights
        """
        self.logger.info(f"Calculating target weights for {current_date.strftime('%Y-%m-%d')}")
        
        # Step 1: Get price data for all assets
        price_data = self._get_price_data(current_date)
        
        # Step 2: Calculate returns data
        returns_data = self._calculate_returns_data(price_data)
        
        # Step 3: Calculate momentum signals
        momentum_signals = self._calculate_momentum_signals(price_data, current_date)
        
        # Step 4: Calculate covariance matrix
        covariance_matrix = self._calculate_covariance_matrix(returns_data)
        
        # Step 5: Convert momentum signals to expected returns
        expected_returns = self._momentum_to_expected_returns(momentum_signals)
        
        # Step 6: Optimize portfolio weights
        constraints = self._get_weight_constraints()
        optimal_weights = self.portfolio_optimizer.optimize_weights(
            expected_returns, covariance_matrix, constraints
        )
        
        # Step 7: Apply volatility targeting
        portfolio_vol = self.risk_calculator.calculate_portfolio_volatility(
            optimal_weights, covariance_matrix
        )
        
        target_weights, leverage = self.portfolio_optimizer.apply_volatility_targeting(
            optimal_weights, self.config.target_volatility, portfolio_vol
        )
        
        self.logger.info(f"Target weights calculated with leverage {leverage:.3f}")
        return target_weights
    
    def _get_price_data(self, current_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get price data for all assets in universe."""
        # Calculate required lookback period
        max_lookback = max(self.config.momentum_lookback_periods)
        lookback_days = max_lookback * 22 + 100  # Add buffer for weekends/holidays
        start_date = current_date - timedelta(days=lookback_days)
        
        price_data = {}
        symbols = self.asset_universe.get_symbols()
        
        for symbol in symbols:
            try:
                data = self.data_provider.get_price_data(symbol, start_date, current_date)
                if len(data) > 0:
                    price_data[symbol] = data
                    self.logger.debug(f"Retrieved {len(data)} days of data for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {str(e)}")
        
        if len(price_data) < 2:
            raise BacktestError("Insufficient price data for strategy execution")
        
        return price_data
    
    def _calculate_returns_data(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns data for all assets."""
        returns_dict = {}
        
        for symbol, data in price_data.items():
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                returns_dict[symbol] = returns
        
        if not returns_dict:
            raise BacktestError("No valid returns data calculated")
        
        # Combine into single DataFrame with aligned dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Remove dates with missing data for any asset
        
        self.logger.debug(f"Calculated returns data: {len(returns_df)} observations for {len(returns_df.columns)} assets")
        return returns_df
    
    def _calculate_momentum_signals(self, price_data: Dict[str, pd.DataFrame], 
                                  current_date: datetime) -> Dict[str, float]:
        """Calculate momentum signals for all assets."""
        # Convert monthly lookback periods to daily
        lookback_periods_daily = [period * 21 for period in self.config.momentum_lookback_periods]
        
        momentum_signals = self.momentum_calculator.calculate_momentum_signals(
            price_data, lookback_periods_daily
        )
        
        self.logger.info(f"Calculated momentum signals for {len(momentum_signals)} assets")
        return momentum_signals
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix from returns data."""
        # Use exponentially weighted covariance for more responsive estimates
        ewm_span = min(252, len(returns_data))  # 1 year or available data
        cov_matrix = returns_data.ewm(span=ewm_span).cov().iloc[-len(returns_data.columns):] * 252  # Annualized
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
        eigenvals = np.maximum(eigenvals, 0.001)  # Floor eigenvalues
        cov_matrix_pd = pd.DataFrame(
            eigenvecs @ np.diag(eigenvals) @ eigenvecs.T,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )
        
        self.logger.debug(f"Calculated covariance matrix for {len(cov_matrix_pd)} assets")
        return cov_matrix_pd
    
    def _momentum_to_expected_returns(self, momentum_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Convert momentum signals to expected returns.
        
        The ReSolve approach uses momentum signals as proxies for expected returns,
        with appropriate scaling to realistic return levels.
        """
        # Scale momentum signals to reasonable expected return range
        # Typical scaling: momentum signal * base_return_scaling
        base_return_scaling = 0.1  # Scale factor to convert momentum to expected returns
        
        expected_returns = {}
        for symbol, momentum in momentum_signals.items():
            # Apply scaling and add base return assumption
            base_return = 0.08  # 8% base expected return
            momentum_adjustment = momentum * base_return_scaling
            expected_return = base_return + momentum_adjustment
            
            # Cap expected returns to reasonable range
            expected_return = np.clip(expected_return, -0.2, 0.5)  # -20% to +50%
            expected_returns[symbol] = expected_return
        
        self.logger.debug(f"Converted momentum signals to expected returns")
        return expected_returns
    
    def _get_weight_constraints(self) -> Dict[str, Tuple[float, float]]:
        """Get weight constraints for optimization."""
        constraints = {}
        
        for symbol in self.asset_universe.get_symbols():
            constraints[symbol] = (self.config.minimum_weight, self.config.maximum_weight)
        
        return constraints
    
    def should_rebalance(self, current_date: datetime) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_date: Current date
            
        Returns:
            True if rebalancing is needed
        """
        if self.last_rebalance_date is None:
            return True  # First rebalance
        
        # Calculate days since last rebalance
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        
        # Determine rebalancing frequency
        if self.config.rebalancing_frequency == 'daily':
            return days_since_rebalance >= 1
        elif self.config.rebalancing_frequency == 'weekly':
            return days_since_rebalance >= 7
        elif self.config.rebalancing_frequency == 'monthly':
            return days_since_rebalance >= 30
        elif self.config.rebalancing_frequency == 'quarterly':
            return days_since_rebalance >= 90
        else:
            return days_since_rebalance >= 365  # Annual
    
    def rebalance_portfolio(self, current_date: datetime) -> Dict[str, float]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            current_date: Date for rebalancing
            
        Returns:
            New portfolio weights
        """
        self.logger.info(f"Rebalancing portfolio on {current_date.strftime('%Y-%m-%d')}")
        
        # Calculate target weights
        target_weights = self.calculate_target_weights(current_date)
        
        # Update current weights and last rebalance date
        self.current_weights = target_weights.copy()
        self.last_rebalance_date = current_date
        
        self.logger.info(f"Portfolio rebalanced: {len(target_weights)} positions")
        return target_weights
    
    def get_portfolio_analytics(self, current_date: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio analytics.
        
        Args:
            current_date: Date for analytics calculation
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not self.current_weights:
            return {}
        
        try:
            # Get recent price data for analytics
            price_data = self._get_price_data(current_date)
            returns_data = self._calculate_returns_data(price_data)
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns_data)
            
            # Calculate risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(portfolio_returns)
            
            # Calculate additional metrics
            covariance_matrix = self._calculate_covariance_matrix(returns_data)
            portfolio_vol = self.risk_calculator.calculate_portfolio_volatility(
                self.current_weights, covariance_matrix
            )
            
            analytics = {
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'maximum_drawdown': risk_metrics.maximum_drawdown,
                'value_at_risk_95': risk_metrics.value_at_risk_95,
                'calmar_ratio': risk_metrics.calmar_ratio,
                'number_of_positions': len([w for w in self.current_weights.values() if w > 0.001])
            }
            
            self.logger.debug(f"Calculated portfolio analytics: {analytics}")
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio analytics: {str(e)}")
            return {}
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns given weights and asset returns."""
        if not self.current_weights:
            return pd.Series(dtype=float)
        
        # Align weights with available return data
        common_assets = set(self.current_weights.keys()) & set(returns_data.columns)
        if not common_assets:
            return pd.Series(dtype=float)
        
        # Create weight series
        weights = pd.Series(self.current_weights)[list(common_assets)]
        weights = weights / weights.sum()  # Normalize
        
        # Calculate portfolio returns
        aligned_returns = returns_data[list(common_assets)]
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        return portfolio_returns
    
    def get_current_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation."""
        return self.current_weights.copy()
    
    def get_momentum_breakdown(self, current_date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Get detailed momentum breakdown for all assets.
        
        Returns:
            Dictionary with momentum components for each asset
        """
        try:
            price_data = self._get_price_data(current_date)
            
            if hasattr(self.momentum_calculator, 'get_momentum_decomposition'):
                lookback_periods_daily = [period * 21 for period in self.config.momentum_lookback_periods]
                return self.momentum_calculator.get_momentum_decomposition(price_data, lookback_periods_daily)
            else:
                # Fallback: calculate simple momentum signals
                momentum_signals = self._calculate_momentum_signals(price_data, current_date)
                return {symbol: {'combined': score} for symbol, score in momentum_signals.items()}
                
        except Exception as e:
            self.logger.error(f"Failed to get momentum breakdown: {str(e)}")
            return {}
    
    def get_risk_contribution(self, current_date: datetime) -> Dict[str, float]:
        """
        Get risk contribution of each asset to portfolio risk.
        
        Returns:
            Dictionary of risk contributions (sum to 1.0)
        """
        if not self.current_weights:
            return {}
        
        try:
            price_data = self._get_price_data(current_date)
            returns_data = self._calculate_returns_data(price_data)
            covariance_matrix = self._calculate_covariance_matrix(returns_data)
            
            return self.risk_calculator.calculate_risk_contribution(
                self.current_weights, covariance_matrix
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk contribution: {str(e)}")
            return {}
    
    def validate_strategy_state(self) -> bool:
        """
        Validate current strategy state.
        
        Returns:
            True if strategy state is valid
        """
        try:
            # Check if weights sum to approximately 1
            if self.current_weights:
                total_weight = sum(self.current_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    self.logger.warning(f"Portfolio weights sum to {total_weight:.4f}, not 1.0")
                    return False
            
            # Check if all weights are within bounds
            for symbol, weight in self.current_weights.items():
                if weight < 0 or weight > 1:
                    self.logger.warning(f"Invalid weight for {symbol}: {weight}")
                    return False
            
            self.logger.debug("Strategy state validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy state validation failed: {str(e)}")
            return False

