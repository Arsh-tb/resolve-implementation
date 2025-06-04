"""
Risk Calculation Implementation for ReSolve Strategy
Implements comprehensive risk metrics and volatility estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

from ..core.base import IRiskCalculator, RiskMetrics, ValidationError


class ResolveRiskCalculator(IRiskCalculator):
    """
    Implements risk calculation methods used in the ReSolve methodology.
    
    Includes volatility estimation, correlation analysis, and comprehensive
    risk metrics calculation with proper handling of Indian market characteristics.
    """
    
    def __init__(self, 
                 volatility_method: str = 'ewm',
                 correlation_method: str = 'pearson',
                 confidence_levels: Tuple[float, float] = (0.95, 0.99),
                 min_observations: int = 252):
        """
        Initialize risk calculator.
        
        Args:
            volatility_method: Method for volatility calculation ('simple', 'ewm', 'garch')
            correlation_method: Method for correlation calculation ('pearson', 'spearman', 'kendall')
            confidence_levels: Confidence levels for VaR calculation
            min_observations: Minimum observations required for calculations
        """
        self.volatility_method = volatility_method
        self.correlation_method = correlation_method
        self.confidence_levels = confidence_levels
        self.min_observations = min_observations
        self.logger = logging.getLogger(__name__)
        
        # Validate parameters
        if volatility_method not in ['simple', 'ewm', 'garch']:
            raise ValidationError(f"Invalid volatility method: {volatility_method}")
        
        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValidationError(f"Invalid correlation method: {correlation_method}")
    
    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> float:
        """
        Calculate annualized volatility using specified method.
        
        Args:
            returns: Return series (daily returns)
            window: Rolling window for calculation
            
        Returns:
            Annualized volatility
        """
        if len(returns) < self.min_observations:
            self.logger.warning(f"Insufficient data for volatility calculation: {len(returns)} < {self.min_observations}")
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            raise ValidationError("No valid returns data for volatility calculation")
        
        if self.volatility_method == 'simple':
            volatility = returns_clean.std() * np.sqrt(252)
            
        elif self.volatility_method == 'ewm':
            # Exponentially weighted moving average (more responsive to recent data)
            span = min(window, len(returns_clean))
            ewm_var = returns_clean.ewm(span=span).var().iloc[-1]
            volatility = np.sqrt(ewm_var * 252)
            
        elif self.volatility_method == 'garch':
            # Simplified GARCH(1,1) estimation
            volatility = self._calculate_garch_volatility(returns_clean)
        
        else:
            raise ValidationError(f"Unknown volatility method: {self.volatility_method}")
        
        self.logger.debug(f"Calculated volatility: {volatility:.4f}")
        return volatility
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """
        Calculate GARCH(1,1) volatility estimate.
        
        Simplified implementation without external dependencies.
        """
        # Simple GARCH(1,1) parameters (typical values)
        omega = 0.000001  # Long-term variance
        alpha = 0.1       # ARCH parameter
        beta = 0.85       # GARCH parameter
        
        # Initialize variance
        variance = returns.var()
        variances = [variance]
        
        # Calculate conditional variances
        for i in range(1, len(returns)):
            variance = omega + alpha * (returns.iloc[i-1] ** 2) + beta * variance
            variances.append(variance)
        
        # Return annualized volatility
        current_variance = variances[-1]
        return np.sqrt(current_variance * 252)
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple return series.
        
        Args:
            returns_data: DataFrame with return series for multiple assets
            
        Returns:
            Correlation matrix
        """
        if returns_data.empty:
            raise ValidationError("Empty returns data provided")
        
        # Remove columns with insufficient data
        valid_columns = []
        for col in returns_data.columns:
            if returns_data[col].count() >= self.min_observations:
                valid_columns.append(col)
            else:
                self.logger.warning(f"Insufficient data for {col}: {returns_data[col].count()} observations")
        
        if len(valid_columns) < 2:
            raise ValidationError("Need at least 2 assets with sufficient data for correlation matrix")
        
        clean_data = returns_data[valid_columns].dropna()
        
        if len(clean_data) < self.min_observations:
            self.logger.warning(f"Limited overlapping data: {len(clean_data)} observations")
        
        # Calculate correlation matrix
        correlation_matrix = clean_data.corr(method=self.correlation_method)
        
        # Ensure matrix is valid (no NaN values)
        if correlation_matrix.isnull().any().any():
            self.logger.warning("NaN values found in correlation matrix, using pairwise correlations")
            correlation_matrix = clean_data.corr(method=self.correlation_method, min_periods=50)
        
        self.logger.info(f"Calculated correlation matrix for {len(correlation_matrix)} assets")
        return correlation_matrix
    
    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for return series.
        
        Args:
            returns: Return series (daily returns)
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < self.min_observations:
            self.logger.warning(f"Limited data for risk metrics: {len(returns_clean)} observations")
        
        if len(returns_clean) == 0:
            raise ValidationError("No valid returns data for risk metrics calculation")
        
        # Basic statistics
        volatility = self.calculate_volatility(returns_clean)
        mean_return = returns_clean.mean() * 252  # Annualized
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns_clean, (1 - self.confidence_levels[0]) * 100)
        var_99 = np.percentile(returns_clean, (1 - self.confidence_levels[1]) * 100)
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Sharpe Ratio (assuming 6.5% risk-free rate for India)
        risk_free_rate = 0.065
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = float('inf') if excess_return > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        risk_metrics = RiskMetrics(
            volatility=volatility,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            maximum_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
        
        self.logger.debug(f"Calculated risk metrics: Volatility={volatility:.4f}, Sharpe={sharpe_ratio:.4f}")
        return risk_metrics
    
    def calculate_portfolio_volatility(self, weights: Dict[str, float], 
                                     covariance_matrix: pd.DataFrame) -> float:
        """
        Calculate portfolio volatility given weights and covariance matrix.
        
        Args:
            weights: Dictionary of asset weights
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Portfolio volatility (annualized)
        """
        # Ensure weights and covariance matrix are aligned
        common_assets = set(weights.keys()) & set(covariance_matrix.index)
        if len(common_assets) != len(weights):
            missing_assets = set(weights.keys()) - common_assets
            self.logger.warning(f"Missing covariance data for assets: {missing_assets}")
        
        # Create weight vector
        weight_vector = pd.Series(weights)[list(common_assets)]
        weight_vector = weight_vector / weight_vector.sum()  # Normalize
        
        # Subset covariance matrix
        cov_subset = covariance_matrix.loc[list(common_assets), list(common_assets)]
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weight_vector.values, np.dot(cov_subset.values, weight_vector.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        self.logger.debug(f"Portfolio volatility: {portfolio_volatility:.4f}")
        return portfolio_volatility
    
    def calculate_risk_contribution(self, weights: Dict[str, float], 
                                  covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio risk.
        
        Args:
            weights: Dictionary of asset weights
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Dictionary of risk contributions (sum to 1.0)
        """
        common_assets = list(set(weights.keys()) & set(covariance_matrix.index))
        weight_vector = pd.Series(weights)[common_assets]
        weight_vector = weight_vector / weight_vector.sum()
        
        cov_subset = covariance_matrix.loc[common_assets, common_assets]
        
        # Calculate marginal risk contributions
        portfolio_variance = np.dot(weight_vector.values, np.dot(cov_subset.values, weight_vector.values))
        marginal_contributions = np.dot(cov_subset.values, weight_vector.values) / np.sqrt(portfolio_variance)
        
        # Calculate component risk contributions
        risk_contributions = weight_vector.values * marginal_contributions
        risk_contributions = risk_contributions / risk_contributions.sum()  # Normalize
        
        return dict(zip(common_assets, risk_contributions))
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta of asset relative to market.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        # Align series
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < self.min_observations:
            self.logger.warning(f"Limited data for beta calculation: {len(aligned_data)} observations")
        
        if len(aligned_data) == 0:
            raise ValidationError("No overlapping data for beta calculation")
        
        # Calculate beta using linear regression
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        beta = covariance / market_variance if market_variance > 0 else 0
        
        self.logger.debug(f"Calculated beta: {beta:.4f}")
        return beta
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error relative to benchmark.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Annualized tracking error
        """
        # Align series
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) == 0:
            raise ValidationError("No overlapping data for tracking error calculation")
        
        # Calculate excess returns
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        
        # Annualized tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        self.logger.debug(f"Tracking error: {tracking_error:.4f}")
        return tracking_error
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio (excess return / tracking error).
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Information ratio
        """
        # Align series
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) == 0:
            raise ValidationError("No overlapping data for information ratio calculation")
        
        # Calculate excess returns
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        
        # Annualized excess return and tracking error
        excess_return_annual = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        information_ratio = excess_return_annual / tracking_error if tracking_error > 0 else 0
        
        self.logger.debug(f"Information ratio: {information_ratio:.4f}")
        return information_ratio

