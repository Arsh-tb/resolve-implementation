"""
Main execution script for ReSolve Adaptive Asset Allocation Strategy
Demonstrates complete implementation with Indian market data.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.base import StrategyConfiguration, AssetUniverse, AssetData
from src.data.provider import IndianMarketDataProvider
from src.core.strategy import ResolveStrategyEngine
from src.analytics.backtest import ResolveBacktestEngine


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('resolve_strategy.log'),
            logging.StreamHandler()
        ]
    )


def create_indian_asset_universe() -> AssetUniverse:
    """Create asset universe focused on Indian markets."""
    assets = [
        # Indian Equity Indices
        AssetData("NIFTY50", "Nifty 50 Index", "equity", "INR", "NSE"),
        AssetData("NIFTYNEXT50", "Nifty Next 50 Index", "equity", "INR", "NSE"),
        AssetData("NIFTYMIDCAP", "Nifty Midcap 150 Index", "equity", "INR", "NSE"),
        AssetData("NIFTYSMALLCAP", "Nifty Smallcap 250 Index", "equity", "INR", "NSE"),
        
        # Indian Fixed Income
        AssetData("NIFTY10YRGOVT", "Nifty 10 Year Government Bond Index", "fixed_income", "INR", "NSE"),
        AssetData("NIFTYCORPBOND", "Nifty Corporate Bond Index", "fixed_income", "INR", "NSE"),
        AssetData("NIFTYGSEC", "Nifty Government Securities Index", "fixed_income", "INR", "NSE"),
        
        # Indian Real Estate
        AssetData("NIFTYREIT", "Nifty REIT Index", "real_estate", "INR", "NSE"),
        AssetData("NIFTYREALTY", "Nifty Realty Index", "real_estate", "INR", "NSE"),
        
        # Global Diversification (INR hedged)
        AssetData("MSCIWORLD", "MSCI World Index (INR)", "equity", "INR", "Global"),
        AssetData("MSCIEM", "MSCI Emerging Markets Index (INR)", "equity", "INR", "Global"),
        AssetData("GOLD", "Gold (INR)", "commodity", "INR", "Global"),
        AssetData("CRUDE", "Crude Oil (INR)", "commodity", "INR", "Global"),
        
        # Currency
        AssetData("USDINR", "USD/INR Exchange Rate", "currency", "INR", "NSE")
    ]
    
    return AssetUniverse(assets)


def create_strategy_configuration() -> StrategyConfiguration:
    """Create ReSolve strategy configuration."""
    return StrategyConfiguration(
        target_volatility=0.10,  # 10% target volatility
        momentum_lookback_periods=[1, 3, 6, 9, 12],  # 1M, 3M, 6M, 9M, 12M
        rebalancing_frequency='monthly',
        minimum_weight=0.0,
        maximum_weight=0.25,  # Max 25% in any single asset
        transaction_cost=0.001  # 0.1% transaction cost
    )


def run_strategy_demonstration():
    """Run complete strategy demonstration."""
    print("=" * 80)
    print("ReSolve Adaptive Asset Allocation - Indian Markets Implementation")
    print("=" * 80)
    print()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create components
    asset_universe = create_indian_asset_universe()
    config = create_strategy_configuration()
    data_provider = IndianMarketDataProvider()
    
    print(f"Asset Universe: {len(asset_universe.get_symbols())} assets")
    print(f"Target Volatility: {config.target_volatility:.1%}")
    print(f"Momentum Periods: {config.momentum_lookback_periods} months")
    print()
    
    # Initialize strategy engine
    strategy_engine = ResolveStrategyEngine(
        data_provider=data_provider,
        asset_universe=asset_universe,
        config=config
    )
    
    print("Strategy Engine initialized successfully!")
    print()
    
    # Demonstrate current allocation calculation
    current_date = datetime.now()
    print(f"Calculating target allocation for {current_date.strftime('%Y-%m-%d')}...")
    
    try:
        target_weights = strategy_engine.calculate_target_weights(current_date)
        
        print("\nCURRENT TARGET ALLOCATION:")
        print("-" * 50)
        for symbol, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.001:  # Only show meaningful allocations
                asset = asset_universe.get_asset(symbol)
                print(f"{symbol:15} {asset.name:30} {weight:8.2%}")
        
        print(f"\nTotal Allocation: {sum(target_weights.values()):.2%}")
        print()
        
    except Exception as e:
        logger.error(f"Failed to calculate target weights: {str(e)}")
        print(f"Error calculating allocation: {str(e)}")
        print()
    
    # Demonstrate momentum analysis
    print("MOMENTUM ANALYSIS:")
    print("-" * 50)
    
    try:
        momentum_breakdown = strategy_engine.get_momentum_breakdown(current_date)
        
        for symbol, momentum_data in momentum_breakdown.items():
            if 'combined' in momentum_data:
                asset = asset_universe.get_asset(symbol)
                momentum_score = momentum_data['combined']
                print(f"{symbol:15} {asset.name:30} {momentum_score:8.4f}")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to calculate momentum: {str(e)}")
        print(f"Error calculating momentum: {str(e)}")
        print()
    
    # Demonstrate portfolio analytics
    print("PORTFOLIO ANALYTICS:")
    print("-" * 50)
    
    try:
        analytics = strategy_engine.get_portfolio_analytics(current_date)
        
        for metric, value in analytics.items():
            if isinstance(value, float):
                if 'ratio' in metric.lower() or 'volatility' in metric.lower():
                    print(f"{metric:25} {value:8.4f}")
                else:
                    print(f"{metric:25} {value:8.2%}")
            else:
                print(f"{metric:25} {value}")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to calculate analytics: {str(e)}")
        print(f"Error calculating analytics: {str(e)}")
        print()
    
    # Run backtest demonstration
    print("BACKTESTING DEMONSTRATION:")
    print("-" * 50)
    
    try:
        # Create backtest engine
        backtest_engine = ResolveBacktestEngine(
            strategy_engine=strategy_engine,
            transaction_cost=config.transaction_cost,
            initial_capital=10000000  # ₹1 Crore
        )
        
        # Run 1-year backtest
        end_date = current_date
        start_date = end_date - timedelta(days=365)
        
        print(f"Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("This may take a few moments...")
        print()
        
        results = backtest_engine.run_backtest(start_date, end_date)
        
        if 'performance_summary' in results:
            print("BACKTEST RESULTS:")
            print("-" * 30)
            summary = results['performance_summary']
            for metric, value in summary.iterrows():
                print(f"{metric}: {value.iloc[0]}")
            print()
        
        # Generate detailed report
        detailed_report = backtest_engine.generate_detailed_report()
        
        # Save results
        with open('backtest_report.txt', 'w') as f:
            f.write(detailed_report)
        
        if 'portfolio_history' in results:
            results['portfolio_history'].to_csv('portfolio_history.csv')
            print("Detailed results saved to 'portfolio_history.csv'")
        
        if 'weights_history' in results:
            results['weights_history'].to_csv('weights_history.csv')
            print("Weights history saved to 'weights_history.csv'")
        
        print("Detailed report saved to 'backtest_report.txt'")
        print()
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        print(f"Backtest error: {str(e)}")
        print()
    
    # Data requirements information
    print("DATA REQUIREMENTS:")
    print("-" * 50)
    print("This implementation uses synthetic data for demonstration.")
    print("For production use, connect to Bloomberg Terminal with:")
    print()
    print("Required Bloomberg Fields:")
    print("- PX_LAST (Last Price)")
    print("- PX_OPEN (Open Price)")
    print("- PX_HIGH (High Price)")
    print("- PX_LOW (Low Price)")
    print("- PX_VOLUME (Volume)")
    print()
    print("Required Tickers:")
    for symbol in asset_universe.get_symbols():
        asset = asset_universe.get_asset(symbol)
        print(f"- {symbol}: {asset.name}")
    print()
    print("Historical Data: Minimum 2 years of daily data")
    print("Update Frequency: Daily (end of day)")
    print("Currency: All prices in INR")
    print()
    
    print("=" * 80)
    print("ReSolve Strategy Implementation Complete!")
    print("Check the generated CSV files and report for detailed results.")
    print("=" * 80)


def run_custom_backtest():
    """Run custom backtest with user parameters."""
    print("\nCUSTOM BACKTEST CONFIGURATION:")
    print("-" * 50)
    
    try:
        # Get user inputs
        start_year = int(input("Enter start year (e.g., 2020): ") or "2020")
        end_year = int(input("Enter end year (e.g., 2023): ") or "2023")
        initial_capital = float(input("Enter initial capital in INR (e.g., 10000000): ") or "10000000")
        target_vol = float(input("Enter target volatility (e.g., 0.10 for 10%): ") or "0.10")
        
        # Create custom configuration
        config = StrategyConfiguration(
            target_volatility=target_vol,
            momentum_lookback_periods=[1, 3, 6, 9, 12],
            rebalancing_frequency='monthly',
            minimum_weight=0.0,
            maximum_weight=0.25,
            transaction_cost=0.001
        )
        
        # Setup and run
        asset_universe = create_indian_asset_universe()
        data_provider = IndianMarketDataProvider()
        
        strategy_engine = ResolveStrategyEngine(
            data_provider=data_provider,
            asset_universe=asset_universe,
            config=config
        )
        
        backtest_engine = ResolveBacktestEngine(
            strategy_engine=strategy_engine,
            transaction_cost=config.transaction_cost,
            initial_capital=initial_capital
        )
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        print(f"\nRunning custom backtest...")
        results = backtest_engine.run_backtest(start_date, end_date)
        
        if 'performance_summary' in results:
            print("\nCUSTOM BACKTEST RESULTS:")
            print("-" * 40)
            summary = results['performance_summary']
            for metric, value in summary.iterrows():
                print(f"{metric}: {value.iloc[0]}")
        
        # Save custom results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if 'portfolio_history' in results:
            results['portfolio_history'].to_csv(f'custom_backtest_{timestamp}.csv')
            print(f"\nResults saved to 'custom_backtest_{timestamp}.csv'")
        
    except Exception as e:
        print(f"Custom backtest failed: {str(e)}")


if __name__ == "__main__":
    print("ReSolve Adaptive Asset Allocation - Python Implementation")
    print("Choose an option:")
    print("1. Run full demonstration")
    print("2. Run custom backtest")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_strategy_demonstration()
    elif choice == "2":
        run_custom_backtest()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Running full demonstration...")
        run_strategy_demonstration()

