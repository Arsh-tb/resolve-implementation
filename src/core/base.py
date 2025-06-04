"""
ReSolve Adaptive Asset Allocation Implementation
Core Abstract Base Classes and Interfaces

This module defines the fundamental abstractions for the ReSolve methodology
implementation, following clean architecture principles and SOLID design patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class AssetData:
    """Immutable data structure representing asset information."""
    symbol: str
    name: str
    asset_type: str
    currency: str
    exchange: str
    sector: Optional[str] = None
    
    def __post_init__(self):
        """Validate asset data after initialization."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Asset symbol must be a non-empty string")
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Asset name must be a non-empty string")


@dataclass
class PriceData:
    """Immutable data structure for price time series data."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None
    
    def __post_init__(self):
        """Validate price data after initialization."""
        if self.close_price <= 0:
            raise ValueError("Close price must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class PortfolioWeight:
    """Immutable data structure representing portfolio weights."""
    symbol: str
    weight: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate portfolio weight data."""
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1")


@dataclass
class RiskMetrics:
    """Immutable data structure for risk metrics."""
    volatility: float
    value_at_risk_95: float
    value_at_risk_99: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    
    def __post_init__(self):
        """Validate risk metrics."""
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")


class IDataProvider(ABC):
    """Abstract interface for data providers."""
    
    @abstractmethod
    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve price data for a given symbol and date range."""
        pass
    
    @abstractmethod
    def get_asset_info(self, symbol: str) -> AssetData:
        """Retrieve asset information for a given symbol."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class IMomentumCalculator(ABC):
    """Abstract interface for momentum calculation strategies."""
    
    @abstractmethod
    def calculate_momentum_score(self, price_data: pd.DataFrame, lookback_periods: List[int]) -> float:
        """Calculate momentum score for given price data and lookback periods."""
        pass
    
    @abstractmethod
    def calculate_momentum_signals(self, price_data: Dict[str, pd.DataFrame], 
                                 lookback_periods: List[int]) -> Dict[str, float]:
        """Calculate momentum signals for multiple assets."""
        pass


class IRiskCalculator(ABC):
    """Abstract interface for risk calculation strategies."""
    
    @abstractmethod
    def calculate_volatility(self, returns: pd.Series, window: int) -> float:
        """Calculate volatility for given returns series."""
        pass
    
    @abstractmethod
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple return series."""
        pass
    
    @abstractmethod
    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics for return series."""
        pass


class IPortfolioOptimizer(ABC):
    """Abstract interface for portfolio optimization strategies."""
    
    @abstractmethod
    def optimize_weights(self, expected_returns: Dict[str, float], 
                        covariance_matrix: pd.DataFrame,
                        constraints: Dict[str, Union[float, Tuple[float, float]]]) -> Dict[str, float]:
        """Optimize portfolio weights given expected returns and covariance matrix."""
        pass
    
    @abstractmethod
    def apply_volatility_targeting(self, weights: Dict[str, float], 
                                 target_volatility: float,
                                 portfolio_volatility: float) -> Tuple[Dict[str, float], float]:
        """Apply volatility targeting to portfolio weights."""
        pass


class IRebalancingStrategy(ABC):
    """Abstract interface for rebalancing strategies."""
    
    @abstractmethod
    def should_rebalance(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float],
                        last_rebalance_date: datetime,
                        current_date: datetime) -> bool:
        """Determine if portfolio should be rebalanced."""
        pass
    
    @abstractmethod
    def get_rebalancing_frequency(self) -> str:
        """Get the rebalancing frequency (e.g., 'monthly', 'quarterly')."""
        pass


class IPerformanceAnalyzer(ABC):
    """Abstract interface for performance analysis."""
    
    @abstractmethod
    def calculate_portfolio_returns(self, weights_history: pd.DataFrame, 
                                  price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate portfolio returns given weights and price data."""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, portfolio_returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        pass
    
    @abstractmethod
    def generate_performance_report(self, portfolio_returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None) -> str:
        """Generate detailed performance report."""
        pass


class IBacktestEngine(ABC):
    """Abstract interface for backtesting engine."""
    
    @abstractmethod
    def run_backtest(self, start_date: datetime, end_date: datetime,
                    initial_capital: float) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Run complete backtest for given date range."""
        pass
    
    @abstractmethod
    def get_backtest_results(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Get results from the last backtest run."""
        pass


class StrategyConfiguration:
    """Configuration class for ReSolve strategy parameters."""
    
    def __init__(self,
                 target_volatility: float = 0.10,
                 momentum_lookback_periods: List[int] = None,
                 rebalancing_frequency: str = "monthly",
                 minimum_weight: float = 0.05,
                 maximum_weight: float = 0.40,
                 transaction_cost: float = 0.001,
                 risk_free_rate: float = 0.065):
        """
        Initialize strategy configuration.
        
        Args:
            target_volatility: Target portfolio volatility (default 10%)
            momentum_lookback_periods: List of lookback periods in months
            rebalancing_frequency: How often to rebalance ('monthly', 'quarterly', etc.)
            minimum_weight: Minimum weight per asset
            maximum_weight: Maximum weight per asset
            transaction_cost: Transaction cost as percentage
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.target_volatility = target_volatility
        self.momentum_lookback_periods = momentum_lookback_periods or [1, 3, 6, 12]
        self.rebalancing_frequency = rebalancing_frequency
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        if not 0.01 <= self.target_volatility <= 0.50:
            raise ValueError("Target volatility must be between 1% and 50%")
        
        if not 0 <= self.minimum_weight <= self.maximum_weight <= 1:
            raise ValueError("Invalid weight constraints")
        
        if self.rebalancing_frequency not in ["daily", "weekly", "monthly", "quarterly", "annually"]:
            raise ValueError("Invalid rebalancing frequency")
        
        if not all(period > 0 for period in self.momentum_lookback_periods):
            raise ValueError("All momentum lookback periods must be positive")


class AssetUniverse:
    """Manages the universe of assets for the strategy."""
    
    def __init__(self, assets: List[AssetData]):
        """Initialize asset universe with list of assets."""
        self.assets = {asset.symbol: asset for asset in assets}
        self._validate_universe()
    
    def _validate_universe(self):
        """Validate asset universe."""
        if len(self.assets) < 2:
            raise ValueError("Asset universe must contain at least 2 assets")
        
        symbols = list(self.assets.keys())
        if len(symbols) != len(set(symbols)):
            raise ValueError("Duplicate symbols found in asset universe")
    
    def get_asset(self, symbol: str) -> AssetData:
        """Get asset data for given symbol."""
        if symbol not in self.assets:
            raise KeyError(f"Asset {symbol} not found in universe")
        return self.assets[symbol]
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in universe."""
        return list(self.assets.keys())
    
    def filter_by_type(self, asset_type: str) -> List[str]:
        """Filter assets by type."""
        return [symbol for symbol, asset in self.assets.items() 
                if asset.asset_type == asset_type]
    
    def filter_by_sector(self, sector: str) -> List[str]:
        """Filter assets by sector."""
        return [symbol for symbol, asset in self.assets.items() 
                if asset.sector == sector]


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataError(Exception):
    """Custom exception for data-related errors."""
    pass


class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass


class BacktestError(Exception):
    """Custom exception for backtesting errors."""
    pass

