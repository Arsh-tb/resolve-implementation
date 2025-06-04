"""
Data Provider Implementation for Indian Markets
Provides synthetic data for ReSolve strategy backtesting with realistic Indian market characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from ..core.base import IDataProvider, AssetData, PriceData, DataError


class IndianMarketDataProvider(IDataProvider):
    """
    Data provider for Indian market assets with synthetic but realistic data.
    Implements proper correlation structures and volatility patterns for Indian assets.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Indian market data provider.
        
        Args:
            random_seed: Seed for reproducible random data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.logger = logging.getLogger(__name__)
        self._initialize_asset_universe()
        self._initialize_market_parameters()
        
    def _initialize_asset_universe(self):
        """Initialize the Indian market asset universe."""
        self.assets = {
            'NIFTY50': AssetData(
                symbol='NIFTY50',
                name='Nifty 50 Index',
                asset_type='equity',
                currency='INR',
                exchange='NSE',
                sector='Large Cap'
            ),
            'NIFTYNEXT50': AssetData(
                symbol='NIFTYNEXT50', 
                name='Nifty Next 50 Index',
                asset_type='equity',
                currency='INR',
                exchange='NSE',
                sector='Large Cap'
            ),
            'NIFTYMIDCAP': AssetData(
                symbol='NIFTYMIDCAP',
                name='Nifty Midcap 100 Index', 
                asset_type='equity',
                currency='INR',
                exchange='NSE',
                sector='Mid Cap'
            ),
            'NIFTYSMALLCAP': AssetData(
                symbol='NIFTYSMALLCAP',
                name='Nifty Smallcap 100 Index',
                asset_type='equity', 
                currency='INR',
                exchange='NSE',
                sector='Small Cap'
            ),
            'INDIABOND7Y': AssetData(
                symbol='INDIABOND7Y',
                name='India 7-Year Government Bond',
                asset_type='bond',
                currency='INR',
                exchange='NSE',
                sector='Government'
            ),
            'INDIABOND10Y': AssetData(
                symbol='INDIABOND10Y', 
                name='India 10-Year Government Bond',
                asset_type='bond',
                currency='INR',
                exchange='NSE',
                sector='Government'
            ),
            'INDIACORPBOND': AssetData(
                symbol='INDIACORPBOND',
                name='India Corporate Bond Index',
                asset_type='bond',
                currency='INR',
                exchange='NSE', 
                sector='Corporate'
            ),
            'NIFTYREITS': AssetData(
                symbol='NIFTYREITS',
                name='Nifty REITs & InvITs Index',
                asset_type='reit',
                currency='INR',
                exchange='NSE',
                sector='Real Estate'
            ),
            'GOLD': AssetData(
                symbol='GOLD',
                name='Gold (INR)',
                asset_type='commodity',
                currency='INR',
                exchange='MCX',
                sector='Precious Metals'
            ),
            'SILVER': AssetData(
                symbol='SILVER',
                name='Silver (INR)', 
                asset_type='commodity',
                currency='INR',
                exchange='MCX',
                sector='Precious Metals'
            ),
            'CRUDEOIL': AssetData(
                symbol='CRUDEOIL',
                name='Crude Oil (INR)',
                asset_type='commodity',
                currency='INR', 
                exchange='MCX',
                sector='Energy'
            ),
            'MSCIWORLD': AssetData(
                symbol='MSCIWORLD',
                name='MSCI World Index (INR Hedged)',
                asset_type='equity',
                currency='INR',
                exchange='International',
                sector='Global Equity'
            ),
            'MSCIEM': AssetData(
                symbol='MSCIEM',
                name='MSCI Emerging Markets (INR Hedged)',
                asset_type='equity',
                currency='INR',
                exchange='International', 
                sector='Emerging Markets'
            ),
            'USDINR': AssetData(
                symbol='USDINR',
                name='USD/INR Currency',
                asset_type='currency',
                currency='INR',
                exchange='NSE',
                sector='Currency'
            )
        }
        
    def _initialize_market_parameters(self):
        """Initialize realistic market parameters for Indian assets."""
        # Expected annual returns (realistic for Indian markets)
        self.expected_returns = {
            'NIFTY50': 0.12,
            'NIFTYNEXT50': 0.14,
            'NIFTYMIDCAP': 0.16,
            'NIFTYSMALLCAP': 0.18,
            'INDIABOND7Y': 0.065,
            'INDIABOND10Y': 0.070,
            'INDIACORPBOND': 0.075,
            'NIFTYREITS': 0.10,
            'GOLD': 0.08,
            'SILVER': 0.09,
            'CRUDEOIL': 0.06,
            'MSCIWORLD': 0.10,
            'MSCIEM': 0.11,
            'USDINR': 0.02
        }
        
        # Annual volatilities (realistic for Indian markets)
        self.volatilities = {
            'NIFTY50': 0.18,
            'NIFTYNEXT50': 0.20,
            'NIFTYMIDCAP': 0.25,
            'NIFTYSMALLCAP': 0.30,
            'INDIABOND7Y': 0.08,
            'INDIABOND10Y': 0.09,
            'INDIACORPBOND': 0.10,
            'NIFTYREITS': 0.22,
            'GOLD': 0.15,
            'SILVER': 0.25,
            'CRUDEOIL': 0.35,
            'MSCIWORLD': 0.16,
            'MSCIEM': 0.22,
            'USDINR': 0.06
        }
        
        # Correlation matrix (realistic correlations for Indian markets)
        symbols = list(self.assets.keys())
        n_assets = len(symbols)
        
        # Create realistic correlation matrix
        correlation_matrix = np.eye(n_assets)
        
        # Define correlation groups
        equity_indices = ['NIFTY50', 'NIFTYNEXT50', 'NIFTYMIDCAP', 'NIFTYSMALLCAP']
        bond_indices = ['INDIABOND7Y', 'INDIABOND10Y', 'INDIACORPBOND']
        commodity_indices = ['GOLD', 'SILVER', 'CRUDEOIL']
        international_indices = ['MSCIWORLD', 'MSCIEM']
        
        # Set correlations within groups
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    if symbol1 in equity_indices and symbol2 in equity_indices:
                        correlation_matrix[i, j] = np.random.uniform(0.7, 0.9)
                    elif symbol1 in bond_indices and symbol2 in bond_indices:
                        correlation_matrix[i, j] = np.random.uniform(0.6, 0.8)
                    elif symbol1 in commodity_indices and symbol2 in commodity_indices:
                        correlation_matrix[i, j] = np.random.uniform(0.3, 0.6)
                    elif symbol1 in international_indices and symbol2 in international_indices:
                        correlation_matrix[i, j] = np.random.uniform(0.8, 0.9)
                    # Cross-group correlations
                    elif (symbol1 in equity_indices and symbol2 in bond_indices) or \
                         (symbol1 in bond_indices and symbol2 in equity_indices):
                        correlation_matrix[i, j] = np.random.uniform(-0.2, 0.1)
                    elif (symbol1 in equity_indices and symbol2 == 'GOLD') or \
                         (symbol1 == 'GOLD' and symbol2 in equity_indices):
                        correlation_matrix[i, j] = np.random.uniform(0.1, 0.3)
                    elif symbol1 == 'USDINR' or symbol2 == 'USDINR':
                        correlation_matrix[i, j] = np.random.uniform(-0.1, 0.2)
                    else:
                        correlation_matrix[i, j] = np.random.uniform(-0.1, 0.3)
        
        # Ensure matrix is symmetric and positive definite
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Make positive definite if needed
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.correlation_matrix = pd.DataFrame(
            correlation_matrix, 
            index=symbols, 
            columns=symbols
        )
        
        self.logger.info(f"Initialized {len(self.assets)} Indian market assets")
        
    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate realistic price data for given symbol and date range.
        
        Args:
            symbol: Asset symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol not in self.assets:
            raise DataError(f"Symbol {symbol} not found in asset universe")
            
        if start_date >= end_date:
            raise DataError("Start date must be before end date")
            
        # Generate date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        if n_days == 0:
            raise DataError("No business days in specified date range")
            
        # Get asset parameters
        annual_return = self.expected_returns[symbol]
        annual_volatility = self.volatilities[symbol]
        
        # Convert to daily parameters
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate random returns with realistic patterns
        np.random.seed(self.random_seed + hash(symbol) % 1000)
        
        # Add momentum and mean reversion patterns
        returns = []
        momentum_factor = 0.1  # Momentum persistence
        mean_reversion_factor = 0.05  # Mean reversion strength
        
        for i in range(n_days):
            # Base random return
            base_return = np.random.normal(daily_return, daily_volatility)
            
            # Add momentum (recent returns influence current return)
            if i > 0 and len(returns) > 0:
                recent_momentum = np.mean(returns[-5:]) if len(returns) >= 5 else returns[-1]
                momentum_adjustment = momentum_factor * recent_momentum
                base_return += momentum_adjustment
            
            # Add mean reversion (extreme returns tend to reverse)
            if i > 0 and len(returns) > 0:
                if abs(returns[-1]) > 2 * daily_volatility:  # Extreme return
                    mean_reversion_adjustment = -mean_reversion_factor * returns[-1]
                    base_return += mean_reversion_adjustment
            
            # Add volatility clustering
            if i > 0 and len(returns) > 0:
                volatility_multiplier = 1 + 0.3 * abs(returns[-1]) / daily_volatility
                base_return *= volatility_multiplier
                
            returns.append(base_return)
        
        returns = np.array(returns)
        
        # Generate price series
        initial_price = 1000 if symbol.startswith('NIFTY') else \
                       100 if symbol.endswith('BOND') else \
                       50000 if symbol == 'GOLD' else \
                       70000 if symbol == 'SILVER' else \
                       5000 if symbol == 'CRUDEOIL' else \
                       3000 if symbol.startswith('MSCI') else \
                       83 if symbol == 'USDINR' else \
                       500
        
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate OHLC data
        ohlc_data = []
        for i, (date, close_price) in enumerate(zip(date_range, prices)):
            # Generate realistic OHLC based on daily volatility
            daily_range = close_price * daily_volatility * np.random.uniform(0.5, 2.0)
            
            open_price = close_price * (1 + np.random.normal(0, daily_volatility * 0.3))
            high_price = max(open_price, close_price) + daily_range * np.random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - daily_range * np.random.uniform(0, 0.5)
            
            # Ensure logical OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume (higher volume on higher volatility days)
            base_volume = 1000000 if symbol.startswith('NIFTY') else 100000
            volume_multiplier = 1 + abs(returns[i]) / daily_volatility
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
            ohlc_data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2), 
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'adjusted_close': round(close_price, 2)
            })
        
        df = pd.DataFrame(ohlc_data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.debug(f"Generated {len(df)} days of price data for {symbol}")
        return df
    
    def get_asset_info(self, symbol: str) -> AssetData:
        """Get asset information for given symbol."""
        if symbol not in self.assets:
            raise DataError(f"Symbol {symbol} not found in asset universe")
        return self.assets[symbol]
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self.assets.keys())
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get the correlation matrix for all assets."""
        return self.correlation_matrix.copy()
    
    def get_expected_returns(self) -> Dict[str, float]:
        """Get expected returns for all assets."""
        return self.expected_returns.copy()
    
    def get_volatilities(self) -> Dict[str, float]:
        """Get volatilities for all assets.""" 
        return self.volatilities.copy()
    
    def validate_data_quality(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate data quality for given symbol and date range.
        
        Returns:
            True if data quality is acceptable, False otherwise
        """
        try:
            data = self.get_price_data(symbol, start_date, end_date)
            
            # Check for missing data
            if data.isnull().any().any():
                self.logger.warning(f"Missing data found for {symbol}")
                return False
            
            # Check for negative prices
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                self.logger.warning(f"Non-positive prices found for {symbol}")
                return False
            
            # Check for logical OHLC relationships
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ).any()
            
            if invalid_ohlc:
                self.logger.warning(f"Invalid OHLC relationships found for {symbol}")
                return False
                
            self.logger.info(f"Data quality validation passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed for {symbol}: {str(e)}")
            return False

