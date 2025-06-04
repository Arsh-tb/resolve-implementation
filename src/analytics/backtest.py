"""
Backtesting Engine for ReSolve Adaptive Asset Allocation Strategy
Comprehensive backtesting framework with performance analysis and reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from ..core.base import IBacktestEngine, IPerformanceAnalyzer, BacktestError
from ..core.strategy import ResolveStrategyEngine


class ResolveBacktestEngine(IBacktestEngine):
    """
    Comprehensive backtesting engine for ReSolve strategy.
    
    Handles:
    1. Historical simulation with proper look-ahead bias prevention
    2. Transaction cost modeling
    3. Performance tracking and analysis
    4. Risk monitoring
    5. Detailed reporting
    """
    
    def __init__(self, 
                 strategy_engine: ResolveStrategyEngine,
                 transaction_cost: float = 0.001,
                 initial_capital: float = 1000000):
        """
        Initialize backtesting engine.
        
        Args:
            strategy_engine: ReSolve strategy engine
            transaction_cost: Transaction cost as percentage of trade value
            initial_capital: Initial capital for backtesting
        """
        self.strategy_engine = strategy_engine
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Backtest state
        self.backtest_results = {}
        self.portfolio_history = []
        self.weights_history = []
        self.returns_history = []
        self.rebalance_dates = []
        
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, start_date: datetime, end_date: datetime,
                    initial_capital: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete backtest for given date range.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital (uses default if None)
            
        Returns:
            Dictionary containing backtest results
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            
        self.logger.info(f"Starting backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Initialize backtest state
        self._initialize_backtest()
        
        # Generate business day range
        business_days = pd.bdate_range(start=start_date, end=end_date, freq='D')
        
        current_capital = self.initial_capital
        current_weights = {}
        
        for current_date in business_days:
            try:
                # Check if rebalancing is needed
                if self.strategy_engine.should_rebalance(current_date):
                    self.logger.debug(f"Rebalancing on {current_date.strftime('%Y-%m-%d')}")
                    
                    # Calculate new target weights
                    new_weights = self.strategy_engine.rebalance_portfolio(current_date)
                    
                    # Calculate transaction costs
                    transaction_costs = self._calculate_transaction_costs(
                        current_weights, new_weights, current_capital
                    )
                    
                    # Apply transaction costs
                    current_capital -= transaction_costs
                    current_weights = new_weights.copy()
                    
                    # Record rebalance
                    self.rebalance_dates.append(current_date)
                    self.weights_history.append({
                        'date': current_date,
                        'weights': current_weights.copy(),
                        'transaction_costs': transaction_costs
                    })
                
                # Calculate daily portfolio return
                daily_return = self._calculate_daily_return(current_date, current_weights)
                
                # Update capital
                current_capital *= (1 + daily_return)
                
                # Record portfolio state
                self.portfolio_history.append({
                    'date': current_date,
                    'capital': current_capital,
                    'daily_return': daily_return,
                    'weights': current_weights.copy()
                })
                
                self.returns_history.append(daily_return)
                
            except Exception as e:
                self.logger.error(f"Error on {current_date.strftime('%Y-%m-%d')}: {str(e)}")
                continue
        
        # Compile results
        self.backtest_results = self._compile_results()
        
        self.logger.info(f"Backtest completed: {len(self.portfolio_history)} days simulated")
        return self.backtest_results
    
    def _initialize_backtest(self):
        """Initialize backtest state."""
        self.portfolio_history = []
        self.weights_history = []
        self.returns_history = []
        self.rebalance_dates = []
        self.backtest_results = {}
        
        # Reset strategy engine state
        self.strategy_engine.current_weights = {}
        self.strategy_engine.last_rebalance_date = None
    
    def _calculate_transaction_costs(self, old_weights: Dict[str, float], 
                                   new_weights: Dict[str, float], 
                                   capital: float) -> float:
        """Calculate transaction costs for rebalancing."""
        if not old_weights:  # First allocation
            return sum(new_weights.values()) * capital * self.transaction_cost
        
        # Calculate turnover
        total_turnover = 0
        all_assets = set(old_weights.keys()) | set(new_weights.keys())
        
        for asset in all_assets:
            old_weight = old_weights.get(asset, 0)
            new_weight = new_weights.get(asset, 0)
            turnover = abs(new_weight - old_weight)
            total_turnover += turnover
        
        # Transaction costs = turnover * capital * cost rate
        transaction_costs = total_turnover * capital * self.transaction_cost
        
        self.logger.debug(f"Transaction costs: {transaction_costs:.2f} (turnover: {total_turnover:.4f})")
        return transaction_costs
    
    def _calculate_daily_return(self, current_date: datetime, 
                              weights: Dict[str, float]) -> float:
        """Calculate daily portfolio return."""
        if not weights:
            return 0.0
        
        try:
            # Get price data for current and previous day
            previous_date = current_date - timedelta(days=1)
            
            daily_returns = {}
            for symbol in weights.keys():
                try:
                    # Get price data
                    price_data = self.strategy_engine.data_provider.get_price_data(
                        symbol, previous_date - timedelta(days=5), current_date
                    )
                    
                    if len(price_data) >= 2:
                        # Calculate daily return
                        current_price = price_data['close'].iloc[-1]
                        previous_price = price_data['close'].iloc[-2]
                        daily_return = (current_price / previous_price) - 1
                        daily_returns[symbol] = daily_return
                    else:
                        daily_returns[symbol] = 0.0
                        
                except Exception as e:
                    self.logger.debug(f"Failed to get return for {symbol}: {str(e)}")
                    daily_returns[symbol] = 0.0
            
            # Calculate weighted portfolio return
            portfolio_return = sum(weights.get(symbol, 0) * daily_returns.get(symbol, 0) 
                                 for symbol in weights.keys())
            
            return portfolio_return
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate daily return: {str(e)}")
            return 0.0
    
    def _compile_results(self) -> Dict[str, pd.DataFrame]:
        """Compile backtest results into organized DataFrames."""
        results = {}
        
        # Portfolio value history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)
            results['portfolio_history'] = portfolio_df
        
        # Weights history
        if self.weights_history:
            weights_data = []
            for entry in self.weights_history:
                row = {'date': entry['date'], 'transaction_costs': entry['transaction_costs']}
                row.update(entry['weights'])
                weights_data.append(row)
            
            weights_df = pd.DataFrame(weights_data)
            weights_df.set_index('date', inplace=True)
            results['weights_history'] = weights_df
        
        # Returns series
        if self.returns_history and self.portfolio_history:
            returns_df = pd.DataFrame({
                'date': [entry['date'] for entry in self.portfolio_history],
                'daily_return': self.returns_history
            })
            returns_df.set_index('date', inplace=True)
            results['returns_history'] = returns_df
        
        # Performance summary
        results['performance_summary'] = self._calculate_performance_summary()
        
        return results
    
    def _calculate_performance_summary(self) -> pd.DataFrame:
        """Calculate comprehensive performance summary."""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        returns = np.array(self.returns_history)
        final_capital = self.portfolio_history[-1]['capital']
        
        # Basic performance metrics
        total_return = (final_capital / self.initial_capital) - 1
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.065  # 6.5% for India
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        capital_series = [entry['capital'] for entry in self.portfolio_history]
        capital_series = np.array(capital_series)
        running_max = np.maximum.accumulate(capital_series)
        drawdowns = (capital_series - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Win rate
        positive_days = np.sum(returns > 0)
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else float('inf')
        
        # Transaction costs
        total_transaction_costs = sum(entry.get('transaction_costs', 0) for entry in self.weights_history)
        transaction_cost_pct = total_transaction_costs / self.initial_capital
        
        # Number of rebalances
        n_rebalances = len(self.rebalance_dates)
        
        summary_data = {
            'Total Return': [f"{total_return:.2%}"],
            'Annualized Return': [f"{annualized_return:.2%}"],
            'Volatility': [f"{volatility:.2%}"],
            'Sharpe Ratio': [f"{sharpe_ratio:.3f}"],
            'Sortino Ratio': [f"{sortino_ratio:.3f}"],
            'Calmar Ratio': [f"{calmar_ratio:.3f}"],
            'Maximum Drawdown': [f"{max_drawdown:.2%}"],
            'Win Rate': [f"{win_rate:.2%}"],
            'VaR (95%)': [f"{var_95:.2%}"],
            'VaR (99%)': [f"{var_99:.2%}"],
            'Total Transaction Costs': [f"{transaction_cost_pct:.2%}"],
            'Number of Rebalances': [n_rebalances],
            'Final Capital': [f"₹{final_capital:,.0f}"]
        }
        
        summary_df = pd.DataFrame(summary_data, index=['Value'])
        return summary_df.T
    
    def get_backtest_results(self) -> Dict[str, pd.DataFrame]:
        """Get results from the last backtest run."""
        return self.backtest_results
    
    def compare_to_benchmark(self, benchmark_symbol: str) -> pd.DataFrame:
        """
        Compare strategy performance to benchmark.
        
        Args:
            benchmark_symbol: Symbol for benchmark comparison
            
        Returns:
            DataFrame with comparative metrics
        """
        if 'returns_history' not in self.backtest_results:
            raise BacktestError("No backtest results available for comparison")
        
        try:
            # Get benchmark data
            start_date = self.backtest_results['returns_history'].index[0]
            end_date = self.backtest_results['returns_history'].index[-1]
            
            benchmark_data = self.strategy_engine.data_provider.get_price_data(
                benchmark_symbol, start_date - timedelta(days=5), end_date
            )
            
            benchmark_returns = benchmark_data['close'].pct_change().dropna()
            
            # Align dates
            strategy_returns = self.backtest_results['returns_history']['daily_return']
            common_dates = strategy_returns.index.intersection(benchmark_returns.index)
            
            strategy_aligned = strategy_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            # Calculate comparative metrics
            strategy_total = (1 + strategy_aligned).prod() - 1
            benchmark_total = (1 + benchmark_aligned).prod() - 1
            
            strategy_vol = strategy_aligned.std() * np.sqrt(252)
            benchmark_vol = benchmark_aligned.std() * np.sqrt(252)
            
            # Tracking error and information ratio
            excess_returns = strategy_aligned - benchmark_aligned
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            comparison_data = {
                'Strategy': [
                    f"{strategy_total:.2%}",
                    f"{strategy_vol:.2%}",
                    f"{(strategy_total + 1) ** (252 / len(strategy_aligned)) - 1:.2%}"
                ],
                'Benchmark': [
                    f"{benchmark_total:.2%}",
                    f"{benchmark_vol:.2%}",
                    f"{(benchmark_total + 1) ** (252 / len(benchmark_aligned)) - 1:.2%}"
                ],
                'Difference': [
                    f"{strategy_total - benchmark_total:.2%}",
                    f"{strategy_vol - benchmark_vol:.2%}",
                    f"{tracking_error:.2%}"
                ]
            }
            
            comparison_df = pd.DataFrame(
                comparison_data,
                index=['Total Return', 'Volatility', 'Annualized Return']
            )
            
            # Add information ratio
            comparison_df.loc['Information Ratio'] = [f"{information_ratio:.3f}", "N/A", "N/A"]
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Failed to compare to benchmark: {str(e)}")
            return pd.DataFrame()
    
    def generate_detailed_report(self) -> str:
        """Generate detailed backtest report."""
        if not self.backtest_results:
            return "No backtest results available."
        
        report = []
        report.append("=" * 80)
        report.append("ReSolve Adaptive Asset Allocation - Backtest Report")
        report.append("=" * 80)
        report.append("")
        
        # Performance summary
        if 'performance_summary' in self.backtest_results:
            report.append("PERFORMANCE SUMMARY")
            report.append("-" * 40)
            summary = self.backtest_results['performance_summary']
            for metric, value in summary.iterrows():
                report.append(f"{metric}: {value.iloc[0]}")
            report.append("")
        
        # Strategy configuration
        config = self.strategy_engine.config
        report.append("STRATEGY CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Target Volatility: {config.target_volatility:.1%}")
        report.append(f"Momentum Lookback Periods: {config.momentum_lookback_periods}")
        report.append(f"Rebalancing Frequency: {config.rebalancing_frequency}")
        report.append(f"Weight Constraints: {config.minimum_weight:.1%} - {config.maximum_weight:.1%}")
        report.append(f"Transaction Cost: {config.transaction_cost:.3%}")
        report.append("")
        
        # Asset universe
        report.append("ASSET UNIVERSE")
        report.append("-" * 40)
        symbols = self.strategy_engine.asset_universe.get_symbols()
        for symbol in symbols:
            asset = self.strategy_engine.asset_universe.get_asset(symbol)
            report.append(f"{symbol}: {asset.name} ({asset.asset_type})")
        report.append("")
        
        return "\n".join(report)

