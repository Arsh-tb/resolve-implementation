"""
ReSolve Adaptive Asset Allocation - Python Implementation
A comprehensive implementation of the ReSolve methodology for Indian markets.
"""

__version__ = "1.0.0"
__author__ = "ReSolve Implementation Team"
__description__ = "ReSolve Adaptive Asset Allocation for Indian Markets"

# Core imports for easy access
from .core.base import (
    StrategyConfiguration,
    AssetUniverse,
    AssetData,
    IDataProvider,
    IMomentumCalculator,
    IRiskCalculator,
    IPortfolioOptimizer
)

from .core.strategy import ResolveStrategyEngine
from .data.provider import IndianMarketDataProvider
from .analytics.backtest import ResolveBacktestEngine

__all__ = [
    'StrategyConfiguration',
    'AssetUniverse', 
    'AssetData',
    'ResolveStrategyEngine',
    'IndianMarketDataProvider',
    'ResolveBacktestEngine',
    'IDataProvider',
    'IMomentumCalculator',
    'IRiskCalculator',
    'IPortfolioOptimizer'
]

