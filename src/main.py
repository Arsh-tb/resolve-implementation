"""
Main execution script for Asset Allocation Strategies

This script provides a unified interface to run all 4 asset allocation strategies:
1. Equal-Weight Portfolio
2. Volatility-Adjusted Risk Parity
3. Momentum-Based Selection
4. Momentum + Risk Parity Hybrid
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy1 import EqualWeightStrategy
from strategy2 import VolatilityAdjustedStrategy
from strategy3 import MomentumSelectionStrategy
from strategy4 import MomentumRiskParityStrategy
from strategy5 import CorrelationAwareMomentumRiskParityStrategy


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('asset_allocation.log'),
            logging.StreamHandler()
        ]
    )


def create_asset_universe() -> List[str]:
    """
    Asset universe matching Excel columns.
    """
    return [
        "Nifty_10_Year",
        "Nifty_5_year",
        "NIFTY_500",
        "Gold",
        "Silver"
    ]


def generate_synthetic_data(start_date: datetime, end_date: datetime, 
                          asset_universe: List[str]) -> pd.DataFrame:
    """
    Generate synthetic historical data for demonstration.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        asset_universe: List of asset symbols
        
    Returns:
        DataFrame with historical price data
    """
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic data
    data = pd.DataFrame(index=date_range)
    
    # Base prices for different asset classes
    base_prices = {
        "NIFTY500": 15000,      # Equity index
        "NIFTY10YRGOVT": 100,   # Long-term bonds
        "NIFTY5YRGOVT": 100,    # Short-term bonds
        "GOLD": 50000,          # Gold per 10g
        "SILVER": 60000,        # Silver per kg
        "CASH": 1.0             # Cash (stable)
    }
    
    # Volatility characteristics
    volatilities = {
        "NIFTY500": 0.015,      # High volatility (equity)
        "NIFTY10YRGOVT": 0.008, # Medium volatility (long bonds)
        "NIFTY5YRGOVT": 0.005,  # Low volatility (short bonds)
        "GOLD": 0.012,          # Medium-high volatility
        "SILVER": 0.020,        # High volatility
        "CASH": 0.0001          # Very low volatility
    }
    
    # Generate price series for each asset
    for asset in asset_universe:
        if asset in base_prices:
            # Generate random walk with drift
            np.random.seed(hash(asset) % 2**32)  # Consistent seed per asset
            
            # Daily returns with drift and volatility
            drift = 0.0005  # Small positive drift
            daily_returns = np.random.normal(drift, volatilities[asset], len(date_range))
            
            # Convert to price series
            prices = [base_prices[asset]]
            for ret in daily_returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            data[asset] = prices
    
    return data


class AssetAllocationEngine:
    """
    Main engine for running asset allocation strategies.
    """
    
    def __init__(self, asset_universe: List[str]):
        """
        Initialize the Asset Allocation Engine.
        
        Args:
            asset_universe: List of asset symbols
        """
        self.asset_universe = asset_universe
        self.strategies = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize all strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all four strategies."""
        self.strategies = {
            'equal_weight': EqualWeightStrategy(self.asset_universe),
            'volatility_adjusted': VolatilityAdjustedStrategy(self.asset_universe),
            'momentum_selection': MomentumSelectionStrategy(self.asset_universe),
            'momentum_risk_parity': MomentumRiskParityStrategy(self.asset_universe),
            'correlation_aware_momentum_risk_parity': CorrelationAwareMomentumRiskParityStrategy(self.asset_universe)
        }
        
        self.logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def run_strategy(self, strategy_name: str, current_date: datetime, 
                    historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Run a specific strategy and get target weights.
        
        Args:
            strategy_name: Name of the strategy to run
            current_date: Current date for calculation
            historical_data: Historical price data
            
        Returns:
            Dictionary of target weights
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        weights = strategy.calculate_target_weights(current_date, historical_data)
        
        self.logger.info(f"Strategy '{strategy_name}' weights: {weights}")
        return weights
    
    def run_all_strategies(self, current_date: datetime, 
                          historical_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Run all strategies and get their target weights.
        
        Args:
            current_date: Current date for calculation
            historical_data: Historical price data
            
        Returns:
            Dictionary mapping strategy names to their target weights
        """
        results = {}
        
        for strategy_name in self.strategies.keys():
            try:
                weights = self.run_strategy(strategy_name, current_date, historical_data)
                results[strategy_name] = weights
            except Exception as e:
                self.logger.error(f"Error running strategy '{strategy_name}': {str(e)}")
                results[strategy_name] = {}
        
        return results
    
    def get_daily_signals(self, strategy_name: str, current_date: datetime, 
                         historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Get daily signals for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            current_date: Current date
            historical_data: Historical price data
            
        Returns:
            Dictionary of daily portfolio weights
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        signals = strategy.get_daily_signals(current_date, historical_data)
        
        return signals
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, str]:
        """
        Get information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_strategy_info()
    
    def compare_strategies(self, current_date: datetime, 
                          historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare all strategies side by side.
        
        Args:
            current_date: Current date for comparison
            historical_data: Historical price data
            
        Returns:
            DataFrame with comparison of all strategies
        """
        results = self.run_all_strategies(current_date, historical_data)
        
        # Create comparison DataFrame
        comparison_data = {}
        for strategy_name, weights in results.items():
            for asset in self.asset_universe:
                comparison_data[f"{strategy_name}_{asset}"] = weights.get(asset, 0.0)
        
        comparison_df = pd.DataFrame([comparison_data])
        comparison_df.index = [current_date]
        
        return comparison_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("Asset Allocation Strategies - Indian Markets Implementation")
    print("=" * 80)
    print()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create asset universe
    asset_universe = create_asset_universe()
    print(f"Asset Universe: {asset_universe}")
    print()
    
    # Initialize engine
    engine = AssetAllocationEngine(asset_universe)
    
    # Read Excel data
    data_path = os.path.join(os.path.dirname(__file__), '../input_data/Combined_input.xlsx')
    historical_data = pd.read_excel(data_path, parse_dates=['Date'])
    historical_data.set_index('Date', inplace=True)
    print(f"Loaded {len(historical_data)} days of data from Excel.")
    print()
    
    min_days = 252
    if len(historical_data) < min_days:
        print(f"Not enough data to start strategies (need at least {min_days} days, found {len(historical_data)}).")
        return
    
    # --- New: Loop through all dates, rebalance monthly, store weights ---
    rebalance_dates = []
    weights_history = {s: [] for s in engine.strategies.keys()}
    last_rebalance_date = {s: None for s in engine.strategies.keys()}
    current_weights = {s: {asset: 0.0 for asset in asset_universe} for s in engine.strategies.keys()}
    rebalance_weights = {s: [] for s in engine.strategies.keys()}
    rebalance_dates_only = []

    # Only consider dates after 252 days
    date_index = historical_data.index[min_days:]
    for i, current_date in enumerate(date_index):
        hist_slice = historical_data.loc[:current_date]
        is_month_end = (current_date.month != date_index[i-1].month) if i > 0 else True
        for strat_name, strat in engine.strategies.items():
            # Only rebalance on month change or first eligible date
            if last_rebalance_date[strat_name] is None or strat.should_rebalance(current_date, last_rebalance_date[strat_name]):
                if is_month_end:
                    current_weights[strat_name] = strat.calculate_target_weights(current_date, hist_slice)
                    last_rebalance_date[strat_name] = current_date
                    # Store rebalance weights
                    rebalance_weights[strat_name].append(current_weights[strat_name].copy())
                    if strat_name == list(engine.strategies.keys())[0]:
                        rebalance_dates_only.append(current_date)
            # Store weights for this date (for daily drift)
            weights_history[strat_name].append(current_weights[strat_name].copy())
        rebalance_dates.append(current_date)

    # Build daily drift DataFrame
    out_data_daily = {}
    for strat_name in engine.strategies.keys():
        for asset in asset_universe:
            out_data_daily[f"{strat_name}_{asset}"] = [w[asset] for w in weights_history[strat_name]]
    out_df_daily = pd.DataFrame(out_data_daily, index=rebalance_dates)
    out_df_daily.index.name = 'Date'

    # Build rebalance-only DataFrame
    out_data_reb = {}
    for strat_name in engine.strategies.keys():
        for asset in asset_universe:
            out_data_reb[f"{strat_name}_{asset}"] = [w[asset] for w in rebalance_weights[strat_name]]
    out_df_reb = pd.DataFrame(out_data_reb, index=rebalance_dates_only)
    out_df_reb.index.name = 'Date'

    print("\nSTRATEGY WEIGHTS TIME SERIES (monthly rebalancing):")
    print(out_df_reb.tail(12).round(4))
    out_df_reb.to_csv('strategy_weights_rebalance.csv')
    print("Saved rebalance weights to 'strategy_weights_rebalance.csv'")
    print("\nSTRATEGY WEIGHTS TIME SERIES (daily drift):")
    print(out_df_daily.tail(12).round(4))
    out_df_daily.to_csv('strategy_weights_daily.csv')
    print("Saved daily drift weights to 'strategy_weights_daily.csv'")
    print("\nAsset Allocation Strategies Complete!")


def run_single_strategy():
    """Run a single strategy with user input."""
    print("\nSINGLE STRATEGY EXECUTION:")
    print("-" * 50)
    
    # Available strategies
    strategies = {
        '1': 'equal_weight',
        '2': 'volatility_adjusted', 
        '3': 'momentum_selection',
        '4': 'momentum_risk_parity',
        '5': 'correlation_aware_momentum_risk_parity'
    }
    
    print("Available strategies:")
    for key, strategy in strategies.items():
        print(f"  {key}. {strategy.replace('_', ' ').title()}")
    
    choice = input("\nSelect strategy (1-5): ").strip()
    
    if choice not in strategies:
        print("Invalid choice. Using Strategy 1 (Equal Weight).")
        choice = '1'
    
    strategy_name = strategies[choice]
    
    # Setup
    asset_universe = create_asset_universe()
    engine = AssetAllocationEngine(asset_universe)
    
    # Read Excel data
    data_path = os.path.join(os.path.dirname(__file__), '../input_data/Combined_input.xlsx')
    historical_data = pd.read_excel(data_path, parse_dates=['Date'])
    historical_data.set_index('Date', inplace=True)
    min_days = 252
    if len(historical_data) < min_days:
        print(f"Not enough data to start strategies (need at least {min_days} days, found {len(historical_data)}).")
        return
    
    # Get the strategy object
    strategy = engine.strategies[strategy_name]
    
    # Get full weight history
    print(f"\nRunning {strategy_name.replace('_', ' ').title()} Strategy...")
    print("-" * 60)
    
    history = strategy.get_weight_history(historical_data, min_days=min_days)
    full_history = history['full_history']
    rebalance_history = history['rebalance_history']
    
    # Display strategy info
    info = engine.get_strategy_info(strategy_name)
    print(f"Strategy: {info['name']}")
    print(f"Description: {info['description']}")
    print()
    
    # Display recent rebalance weights
    print("RECENT REBALANCE WEIGHTS:")
    print("-" * 40)
    if not rebalance_history.empty:
        recent_rebalance = rebalance_history.iloc[-1]
        for asset, weight in recent_rebalance.items():
            if weight > 0.001:
                print(f"  {asset:15} {weight:8.2%}")
        print(f"  Total Allocation: {recent_rebalance.sum():.2%}")
        print(f"  Rebalance Date: {rebalance_history.index[-1].strftime('%Y-%m-%d')}")
    else:
        print("  No rebalancing data available.")
    
    # Display summary statistics
    print(f"\nHISTORY SUMMARY:")
    print("-" * 40)
    print(f"Full history: {len(full_history)} days")
    print(f"Rebalance history: {len(rebalance_history)} rebalance dates")
    print(f"Date range: {full_history.index[0].strftime('%Y-%m-%d')} to {full_history.index[-1].strftime('%Y-%m-%d')}")
    
    # Show recent full history (last 10 days)
    print(f"\nRECENT FULL HISTORY (last 10 days):")
    print("-" * 40)
    if not full_history.empty:
        print(full_history.tail(10).round(4))
    
    # Transform to long format and add price data
    def transform_to_long_format(weights_df: pd.DataFrame, historical_data: pd.DataFrame, name_suffix: str) -> pd.DataFrame:
        """Transform wide format weights to long format with price data."""
        if weights_df.empty:
            return pd.DataFrame(columns=['date', 'accord_code', 'weight', 'price'])
        
        # Reset index and get the actual column name
        reset_df = weights_df.reset_index()
        date_column = reset_df.columns[0]  # First column after reset_index is the date
        
        # Melt the DataFrame to long format
        long_df = reset_df.melt(
            id_vars=[date_column], 
            var_name='accord_code', 
            value_name='weight'
        ).rename(columns={date_column: 'date'})
        
        # Add price data for each date-asset combination
        prices = []
        for _, row in long_df.iterrows():
            date = row['date']
            asset = row['accord_code']
            try:
                # Get price from historical data
                if date in historical_data.index and asset in historical_data.columns:
                    price = historical_data.loc[date, asset]
                else:
                    price = np.nan
            except (KeyError, IndexError):
                price = np.nan
            prices.append(price)
        
        long_df['price'] = prices
        
        # Remove rows with NaN prices only (keep zero weights)
        long_df = long_df[long_df['price'].notna()]
        
        # Sort by date and accord_code
        long_df = long_df.sort_values(['date', 'accord_code']).reset_index(drop=True)
        
        return long_df
    
    # Transform both histories to long format
    full_long = transform_to_long_format(full_history, historical_data, 'full')
    rebalance_long = transform_to_long_format(rebalance_history, historical_data, 'rebalance')
    
    # Save to CSV files in the new format
    strategy_prefix = strategy_name
    full_filename = f'{strategy_prefix}_full_history_long.csv'
    rebalance_filename = f'{strategy_prefix}_rebalance_history_long.csv'
    
    full_long.to_csv(full_filename, index=False)
    rebalance_long.to_csv(rebalance_filename, index=False)
    
    # Display sample of the new format
    print(f"\nLONG FORMAT OUTPUT SAMPLE (Full History - Last 20 rows):")
    print("-" * 60)
    if not full_long.empty:
        print(full_long.tail(20).round(4))
    
    print(f"\nLONG FORMAT OUTPUT SAMPLE (Rebalance History - Last 15 rows):")
    print("-" * 60)
    if not rebalance_long.empty:
        print(rebalance_long.tail(15).round(4))
    
    print(f"\nSaved full history to '{full_filename}' (long format)")
    print(f"Saved rebalance history to '{rebalance_filename}' (long format)")
    print(f"Total full history records: {len(full_long)}")
    print(f"Total rebalance history records: {len(rebalance_long)}")
    print(f"\n{strategy_name.upper().replace('_', ' ')} STRATEGY COMPLETE!")


if __name__ == "__main__":
    print("Asset Allocation Strategies")
    print("Choose an option:")
    print("1. Run all strategies comparison")
    print("2. Run single strategy")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        run_single_strategy()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Running all strategies comparison...")
        main() 