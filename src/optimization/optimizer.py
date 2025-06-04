"""
Portfolio Optimization Implementation for ReSolve Strategy
Implements mean-variance optimization with momentum signals and volatility targeting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from scipy.optimize import minimize
from scipy.linalg import inv

from ..core.base import IPortfolioOptimizer, OptimizationError, ValidationError


class ResolvePortfolioOptimizer(IPortfolioOptimizer):
    """
    Implements the ReSolve portfolio optimization methodology.
    
    Combines momentum signals with mean-variance optimization and applies
    volatility targeting as described in the ReSolve paper.
    """
    
    def __init__(self, 
                 optimization_method: str = 'mean_variance',
                 risk_aversion: float = 1.0,
                 momentum_scaling: float = 1.0,
                 regularization: float = 0.001):
        """
        Initialize portfolio optimizer.
        
        Args:
            optimization_method: Optimization approach ('mean_variance', 'risk_parity', 'momentum_weighted')
            risk_aversion: Risk aversion parameter for mean-variance optimization
            momentum_scaling: Scaling factor for momentum signals
            regularization: Regularization parameter for covariance matrix
        """
        self.optimization_method = optimization_method
        self.risk_aversion = risk_aversion
        self.momentum_scaling = momentum_scaling
        self.regularization = regularization
        self.logger = logging.getLogger(__name__)
        
        if optimization_method not in ['mean_variance', 'risk_parity', 'momentum_weighted']:
            raise ValidationError(f"Invalid optimization method: {optimization_method}")
    
    def optimize_weights(self, expected_returns: Dict[str, float], 
                        covariance_matrix: pd.DataFrame,
                        constraints: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using specified method.
        
        Args:
            expected_returns: Dictionary of expected returns (momentum-adjusted)
            covariance_matrix: Covariance matrix of asset returns
            constraints: Dictionary of (min_weight, max_weight) constraints per asset
            
        Returns:
            Dictionary of optimized weights
        """
        # Validate inputs
        common_assets = set(expected_returns.keys()) & set(covariance_matrix.index)
        if len(common_assets) == 0:
            raise OptimizationError("No common assets between expected returns and covariance matrix")
        
        assets = list(common_assets)
        n_assets = len(assets)
        
        # Prepare data
        mu = pd.Series(expected_returns)[assets].values
        sigma = covariance_matrix.loc[assets, assets].values
        
        # Add regularization to covariance matrix
        sigma_reg = sigma + self.regularization * np.eye(n_assets)
        
        # Set up constraints
        bounds = []
        for asset in assets:
            if constraints and asset in constraints:
                bounds.append(constraints[asset])
            else:
                bounds.append((0.0, 1.0))  # Default: no short selling, max 100%
        
        # Weights must sum to 1
        weight_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        if self.optimization_method == 'mean_variance':
            weights = self._optimize_mean_variance(mu, sigma_reg, bounds, weight_constraint)
        elif self.optimization_method == 'risk_parity':
            weights = self._optimize_risk_parity(sigma_reg, bounds, weight_constraint)
        elif self.optimization_method == 'momentum_weighted':
            weights = self._optimize_momentum_weighted(mu, sigma_reg, bounds, weight_constraint)
        else:
            raise OptimizationError(f"Unknown optimization method: {self.optimization_method}")
        
        # Convert to dictionary
        weight_dict = dict(zip(assets, weights))
        
        self.logger.info(f"Optimized weights using {self.optimization_method}: {weight_dict}")
        return weight_dict
    
    def _optimize_mean_variance(self, mu: np.ndarray, sigma: np.ndarray, 
                              bounds: List[Tuple[float, float]], 
                              weight_constraint: Dict) -> np.ndarray:
        """
        Optimize using mean-variance approach with momentum-adjusted expected returns.
        """
        n_assets = len(mu)
        
        # Objective function: maximize utility = expected return - (risk_aversion/2) * variance
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights, np.dot(sigma, weights))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[weight_constraint],
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
        
        return result.x
    
    def _optimize_risk_parity(self, sigma: np.ndarray, 
                            bounds: List[Tuple[float, float]], 
                            weight_constraint: Dict) -> np.ndarray:
        """
        Optimize for risk parity (equal risk contribution).
        """
        n_assets = sigma.shape[0]
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
            marginal_contrib = np.dot(sigma, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution (1/n each)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Minimize sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Initial guess: inverse volatility weights
        vol_diag = np.sqrt(np.diag(sigma))
        x0 = (1 / vol_diag) / np.sum(1 / vol_diag)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[weight_constraint],
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            self.logger.warning(f"Risk parity optimization failed: {result.message}")
            return x0  # Return initial guess
        
        return result.x
    
    def _optimize_momentum_weighted(self, mu: np.ndarray, sigma: np.ndarray,
                                  bounds: List[Tuple[float, float]], 
                                  weight_constraint: Dict) -> np.ndarray:
        """
        Optimize using momentum-weighted approach with risk adjustment.
        """
        # Calculate inverse volatility weights
        vol_diag = np.sqrt(np.diag(sigma))
        inv_vol_weights = (1 / vol_diag) / np.sum(1 / vol_diag)
        
        # Scale momentum signals
        momentum_signals = mu * self.momentum_scaling
        
        # Combine momentum with inverse volatility
        # Positive momentum gets higher weight, negative momentum gets lower weight
        momentum_adjusted = np.exp(momentum_signals)  # Always positive
        momentum_weights = momentum_adjusted / np.sum(momentum_adjusted)
        
        # Blend momentum and inverse volatility weights
        blend_factor = 0.7  # 70% momentum, 30% inverse volatility
        blended_weights = blend_factor * momentum_weights + (1 - blend_factor) * inv_vol_weights
        
        # Normalize
        weights = blended_weights / np.sum(blended_weights)
        
        # Apply bounds constraints
        for i, (min_w, max_w) in enumerate(bounds):
            weights[i] = np.clip(weights[i], min_w, max_w)
        
        # Renormalize after applying bounds
        weights = weights / np.sum(weights)
        
        return weights
    
    def apply_volatility_targeting(self, weights: Dict[str, float], 
                                 target_volatility: float,
                                 portfolio_volatility: float) -> Tuple[Dict[str, float], float]:
        """
        Apply volatility targeting to scale portfolio weights.
        
        Args:
            weights: Current portfolio weights
            target_volatility: Target portfolio volatility
            portfolio_volatility: Current portfolio volatility
            
        Returns:
            Tuple of (scaled_weights, leverage_factor)
        """
        if portfolio_volatility <= 0:
            raise ValidationError("Portfolio volatility must be positive")
        
        # Calculate leverage factor
        leverage_factor = target_volatility / portfolio_volatility
        
        # Cap leverage (ReSolve typically caps at 2x)
        max_leverage = 2.0
        leverage_factor = min(leverage_factor, max_leverage)
        
        # Scale weights
        scaled_weights = {asset: weight * leverage_factor for asset, weight in weights.items()}
        
        # If leverage < 1, allocate remainder to cash (represented as reduced weights)
        if leverage_factor < 1.0:
            # Weights are already scaled down, no additional cash allocation needed
            pass
        
        self.logger.info(f"Applied volatility targeting: leverage={leverage_factor:.3f}")
        return scaled_weights, leverage_factor
    
    def calculate_efficient_frontier(self, expected_returns: Dict[str, float],
                                   covariance_matrix: pd.DataFrame,
                                   n_portfolios: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier for given assets.
        
        Args:
            expected_returns: Dictionary of expected returns
            covariance_matrix: Covariance matrix
            n_portfolios: Number of portfolios to calculate
            
        Returns:
            Tuple of (returns, volatilities, weights_matrix)
        """
        assets = list(set(expected_returns.keys()) & set(covariance_matrix.index))
        mu = pd.Series(expected_returns)[assets].values
        sigma = covariance_matrix.loc[assets, assets].values
        n_assets = len(assets)
        
        # Range of target returns
        min_ret = np.min(mu)
        max_ret = np.max(mu)
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        efficient_returns = []
        efficient_volatilities = []
        
        for target_ret in target_returns:
            try:
                # Minimize variance subject to target return constraint
                def objective(weights):
                    return np.dot(weights, np.dot(sigma, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
                    {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_ret}  # Target return
                ]
                
                bounds = [(0, 1) for _ in range(n_assets)]  # No short selling
                x0 = np.ones(n_assets) / n_assets
                
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9, 'disp': False}
                )
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, mu)
                    portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                    
                    efficient_portfolios.append(weights)
                    efficient_returns.append(portfolio_return)
                    efficient_volatilities.append(portfolio_vol)
                    
            except Exception as e:
                self.logger.debug(f"Failed to optimize for target return {target_ret}: {str(e)}")
                continue
        
        if not efficient_portfolios:
            raise OptimizationError("Failed to calculate efficient frontier")
        
        return (np.array(efficient_returns), 
                np.array(efficient_volatilities), 
                np.array(efficient_portfolios))
    
    def calculate_maximum_sharpe_portfolio(self, expected_returns: Dict[str, float],
                                         covariance_matrix: pd.DataFrame,
                                         risk_free_rate: float = 0.065) -> Dict[str, float]:
        """
        Calculate maximum Sharpe ratio portfolio.
        
        Args:
            expected_returns: Dictionary of expected returns
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary of optimal weights
        """
        assets = list(set(expected_returns.keys()) & set(covariance_matrix.index))
        mu = pd.Series(expected_returns)[assets].values
        sigma = covariance_matrix.loc[assets, assets].values
        n_assets = len(assets)
        
        # Excess returns
        excess_returns = mu - risk_free_rate
        
        def objective(weights):
            portfolio_return = np.dot(weights, excess_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            return -sharpe_ratio  # Minimize negative Sharpe ratio
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            raise OptimizationError(f"Maximum Sharpe optimization failed: {result.message}")
        
        weights_dict = dict(zip(assets, result.x))
        self.logger.info(f"Maximum Sharpe portfolio calculated: {weights_dict}")
        return weights_dict
    
    def calculate_minimum_variance_portfolio(self, covariance_matrix: pd.DataFrame,
                                           constraints: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Calculate minimum variance portfolio.
        
        Args:
            covariance_matrix: Covariance matrix
            constraints: Weight constraints per asset
            
        Returns:
            Dictionary of optimal weights
        """
        assets = list(covariance_matrix.index)
        sigma = covariance_matrix.values
        n_assets = len(assets)
        
        def objective(weights):
            return np.dot(weights, np.dot(sigma, weights))
        
        # Set up bounds
        bounds = []
        for asset in assets:
            if constraints and asset in constraints:
                bounds.append(constraints[asset])
            else:
                bounds.append((0.0, 1.0))
        
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            raise OptimizationError(f"Minimum variance optimization failed: {result.message}")
        
        weights_dict = dict(zip(assets, result.x))
        self.logger.info(f"Minimum variance portfolio calculated: {weights_dict}")
        return weights_dict

