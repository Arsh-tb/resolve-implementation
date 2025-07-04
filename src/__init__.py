"""
Asset Allocation Strategies Package

This package contains four different asset allocation strategies for Indian markets:

1. Equal-Weight Portfolio (Monthly Rebalanced)
2. Volatility-Adjusted Risk Parity (60-Day Inverse-Vol Weighting)
3. Momentum-Based Selection (6-Month Return Momentum Strategy)
4. Momentum + Risk Parity Hybrid (Momentum Selection with Volatility Weighting)

Each strategy provides daily signals with monthly rebalancing and outputs portfolio weights.
"""

from .strategy1 import EqualWeightStrategy
from .strategy2 import VolatilityAdjustedStrategy
from .strategy3 import MomentumSelectionStrategy
from .strategy4 import MomentumRiskParityStrategy

__version__ = "1.0.0"
__author__ = "Asset Allocation Team"

__all__ = [
    'EqualWeightStrategy',
    'VolatilityAdjustedStrategy', 
    'MomentumSelectionStrategy',
    'MomentumRiskParityStrategy'
] 