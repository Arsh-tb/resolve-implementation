"""
Strategy 5: Correlation-Aware Momentum + Risk Parity

This strategy extends Strategy 4 by integrating asset correlations into the weighting scheme.
It combines momentum selection with Equal Risk Contribution (ERC) weighting that considers
both individual volatilities and cross-asset correlations to ensure each selected asset
contributes equally to total portfolio risk.

Key Features:
- Momentum-based asset selection (6-month returns > average)
- Equal Risk Contribution (ERC) weighting using covariance matrix
- Correlation-aware risk balancing
- Monthly rebalancing
- No allocation to safe assets (cash) - all weight distributed among selected assets
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class CorrelationAwareMomentumRiskParityStrategy:
    """
    Strategy 5: Correlation-Aware Momentum + Risk Parity
    
    Combines momentum selection with Equal Risk Contribution weighting that
    considers asset correlations for true risk diversification.
    """
    
    def __init__(self, asset_universe: List[str]):
        """
        Initialize the Correlation-Aware Momentum + Risk Parity Strategy.
        
        Args:
            asset_universe: List of asset symbols to consider
        """
        self.asset_universe = asset_universe
        self.logger = logging.getLogger(__name__)
        self.lookback_period = 252  # ~1 year for momentum calculation
        self.volatility_window = 60  # 60 days for volatility and correlation
        self.min_selected_assets = 2  # Minimum assets to select for ERC
        
        self.logger.info(f"Initialized Correlation-Aware Momentum + Risk Parity Strategy with {len(asset_universe)} assets")
    
    def calculate_momentum_scores(self, current_date: datetime, 
                                historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate 6-month momentum scores for all assets.
        
        Args:
            current_date: Current date for calculation
            historical_data: Historical price data
            
        Returns:
            Dictionary of momentum scores (6-month returns)
        """
        momentum_scores = {}
        
        # Get 6 months ago date
        six_months_ago = current_date - timedelta(days=180)
        
        for asset in self.asset_universe:
            if asset not in historical_data.columns:
                momentum_scores[asset] = 0.0
                continue
            
            try:
                # Get price data
                asset_data = historical_data[asset].dropna()
                
                # Find dates
                current_price_data = asset_data[asset_data.index <= current_date]
                if len(current_price_data) == 0:
                    momentum_scores[asset] = 0.0
                    continue
                
                current_price = current_price_data.iloc[-1]
                
                # Find price 6 months ago
                past_price_data = asset_data[asset_data.index <= six_months_ago]
                if len(past_price_data) == 0:
                    momentum_scores[asset] = 0.0
                    continue
                
                past_price = past_price_data.iloc[-1]
                
                # Calculate 6-month return
                if past_price > 0:
                    momentum_score = (current_price / past_price) - 1
                else:
                    momentum_score = 0.0
                
                momentum_scores[asset] = momentum_score
                
            except (IndexError, KeyError, ZeroDivisionError) as e:
                self.logger.warning(f"Error calculating momentum for {asset}: {str(e)}")
                momentum_scores[asset] = 0.0
        
        return momentum_scores
    
    def select_momentum_assets(self, momentum_scores: Dict[str, float]) -> List[str]:
        """
        Select assets with above-average momentum.
        
        Args:
            momentum_scores: Dictionary of momentum scores
            
        Returns:
            List of selected asset symbols
        """
        if not momentum_scores or all(score == 0 for score in momentum_scores.values()):
            return []
        
        # Calculate average momentum
        avg_momentum = np.mean(list(momentum_scores.values()))
        
        # Select assets with above-average momentum
        selected_assets = [
            asset for asset, score in momentum_scores.items() 
            if score > avg_momentum
        ]
        
        # Ensure minimum number of assets
        if len(selected_assets) < self.min_selected_assets:
            # If not enough assets selected, take top performers
            sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            selected_assets = [asset for asset, _ in sorted_assets[:self.min_selected_assets]]
        
        self.logger.info(f"Selected {len(selected_assets)} assets with momentum > {avg_momentum:.4f}: {selected_assets}")
        return selected_assets
    
    def calculate_covariance_matrix(self, current_date: datetime, 
                                  historical_data: pd.DataFrame,
                                  selected_assets: List[str]) -> np.ndarray:
        """
        Calculate covariance matrix for selected assets using rolling window.
        
        Args:
            current_date: Current date for calculation
            historical_data: Historical price data
            selected_assets: List of selected asset symbols
            
        Returns:
            Covariance matrix as numpy array
        """
        if len(selected_assets) < 2:
            return np.array([[1.0]])
        
        # Get data up to current date
        data_slice = historical_data[historical_data.index <= current_date]
        
        # Get the last volatility_window days
        if len(data_slice) < self.volatility_window:
            data_slice = data_slice
        else:
            data_slice = data_slice.iloc[-self.volatility_window:]
        
        # Calculate daily returns for selected assets
        returns_data = {}
        for asset in selected_assets:
            if asset in data_slice.columns:
                prices = data_slice[asset].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    returns_data[asset] = returns
        
        if len(returns_data) < 2:
            # If insufficient data, return identity matrix
            n = len(selected_assets)
            return np.eye(n) * 0.01  # Small variance
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate covariance matrix (annualized)
        cov_matrix = returns_df.cov().values * 252  # Annualize
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return cov_matrix
    
    def solve_equal_risk_contribution(self, cov_matrix: np.ndarray, 
                                    selected_assets: List[str]) -> Dict[str, float]:
        """
        Solve for Equal Risk Contribution (ERC) weights using optimization.
        
        Args:
            cov_matrix: Covariance matrix for selected assets
            selected_assets: List of selected asset symbols
            
        Returns:
            Dictionary of ERC weights
        """
        n_assets = len(selected_assets)
        
        if n_assets == 1:
            return {selected_assets[0]: 1.0}
        
        if n_assets == 0:
            return {}
        
        def risk_contribution_objective(weights):
            """
            Objective function: minimize sum of squared differences in risk contributions.
            """
            weights = np.array(weights)
            
            # Portfolio variance
            portfolio_var = weights.T @ cov_matrix @ weights
            
            if portfolio_var < 1e-10:
                return 1e10  # Large penalty for degenerate cases
            
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Marginal risk contributions
            mrc = (cov_matrix @ weights) / portfolio_vol
            
            # Risk contributions
            rc = weights * mrc
            
            # Target equal risk contribution
            target_rc = portfolio_vol / n_assets
            
            # Objective: minimize sum of squared deviations from target
            objective = np.sum((rc - target_rc) ** 2)
            
            return objective
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds: weights >= 0
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        try:
            # Solve optimization
            result = minimize(
                risk_contribution_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = result.x
                
                # Normalize to ensure sum = 1
                weights = weights / np.sum(weights)
                
                # Create weights dictionary
                erc_weights = {asset: weight for asset, weight in zip(selected_assets, weights)}
                
                self.logger.debug(f"ERC optimization successful. Weights: {erc_weights}")
                return erc_weights
            
            else:
                self.logger.warning(f"ERC optimization failed: {result.message}")
                # Fallback to equal weights
                equal_weight = 1.0 / n_assets
                return {asset: equal_weight for asset in selected_assets}
        
        except Exception as e:
            self.logger.error(f"Error in ERC optimization: {str(e)}")
            # Fallback to equal weights
            equal_weight = 1.0 / n_assets
            return {asset: equal_weight for asset in selected_assets}
    
    def calculate_target_weights(self, current_date: datetime, 
                               historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate target portfolio weights using correlation-aware momentum + risk parity.
        
        Args:
            current_date: Current date for calculation
            historical_data: Historical price data
            
        Returns:
            Dictionary of target weights for each asset
        """
        # Check minimum data requirement
        data_up_to_date = historical_data[historical_data.index <= current_date]
        if len(data_up_to_date) < 252:
            self.logger.warning(f"Insufficient data for strategy (need 252 days, have {len(data_up_to_date)})")
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Step 1: Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(current_date, historical_data)
        
        # Step 2: Select assets with above-average momentum
        selected_assets = self.select_momentum_assets(momentum_scores)
        
        if not selected_assets:
            self.logger.warning("No assets selected based on momentum criteria")
            return {asset: 0.0 for asset in self.asset_universe}
        
        # Step 3: Calculate covariance matrix for selected assets
        cov_matrix = self.calculate_covariance_matrix(current_date, historical_data, selected_assets)
        
        # Step 4: Solve for Equal Risk Contribution weights
        erc_weights = self.solve_equal_risk_contribution(cov_matrix, selected_assets)
        
        # Step 5: Create final portfolio weights
        portfolio_weights = {asset: 0.0 for asset in self.asset_universe}
        
        for asset, weight in erc_weights.items():
            if asset in portfolio_weights:
                portfolio_weights[asset] = weight
        
        # Log allocation details
        allocated_weight = sum(portfolio_weights.values())
        non_zero_assets = [asset for asset, weight in portfolio_weights.items() if weight > 0.001]
        
        self.logger.info(f"Portfolio allocation on {current_date.strftime('%Y-%m-%d')}:")
        self.logger.info(f"  Selected assets: {non_zero_assets}")
        self.logger.info(f"  Total allocated: {allocated_weight:.3f}")
        for asset, weight in portfolio_weights.items():
            if weight > 0.001:
                momentum = momentum_scores.get(asset, 0.0)
                self.logger.info(f"  {asset}: {weight:.3f} (momentum: {momentum:.3f})")
        
        return portfolio_weights
    
    def should_rebalance(self, current_date: datetime, last_rebalance_date: Optional[datetime]) -> bool:
        """
        Determine if portfolio should be rebalanced (monthly).
        
        Args:
            current_date: Current date
            last_rebalance_date: Date of last rebalance
            
        Returns:
            True if should rebalance, False otherwise
        """
        if last_rebalance_date is None:
            return True
        
        # Rebalance monthly (if month has changed)
        return current_date.month != last_rebalance_date.month or current_date.year != last_rebalance_date.year
    
    def get_daily_signals(self, current_date: datetime, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Get daily portfolio signals (same as target weights for this strategy).
        
        Args:
            current_date: Current date
            historical_data: Historical price data
            
        Returns:
            Dictionary of daily portfolio weights
        """
        return self.calculate_target_weights(current_date, historical_data)
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'Correlation-Aware Momentum + Risk Parity',
            'description': 'Selects assets with above-average 6-month momentum and applies Equal Risk Contribution weighting considering correlations',
            'rebalancing': 'Monthly',
            'lookback_momentum': '6 months (180 days)',
            'lookback_volatility': '60 days',
            'risk_model': 'Equal Risk Contribution with correlation matrix',
            'safe_assets': 'None - all weight distributed among selected assets'
        }
    
    def get_weight_history(self, historical_data: pd.DataFrame, min_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Get complete weight history for the strategy.
        
        Args:
            historical_data: Historical price data
            min_days: Minimum days of data required before starting
            
        Returns:
            Dictionary with 'full_history' and 'rebalance_history' DataFrames
        """
        # Only consider dates after min_days
        eligible_dates = historical_data.index[min_days:]
        
        if len(eligible_dates) == 0:
            empty_df = pd.DataFrame(columns=self.asset_universe)
            return {'full_history': empty_df, 'rebalance_history': empty_df}
        
        # Track weights and rebalancing
        all_weights = []
        rebalance_weights = []
        rebalance_dates = []
        last_rebalance_date = None
        current_weights = {asset: 0.0 for asset in self.asset_universe}
        
        for current_date in eligible_dates:
            # Get historical data up to current date
            hist_slice = historical_data.loc[:current_date]
            
            # Check if should rebalance
            if self.should_rebalance(current_date, last_rebalance_date):
                # Calculate new target weights
                current_weights = self.calculate_target_weights(current_date, hist_slice)
                last_rebalance_date = current_date
                
                # Store rebalance weights
                rebalance_weights.append(current_weights.copy())
                rebalance_dates.append(current_date)
            
            # Store current weights (for daily history)
            all_weights.append(current_weights.copy())
        
        # Create DataFrames
        full_history = pd.DataFrame(all_weights, index=eligible_dates)
        
        if rebalance_dates:
            rebalance_history = pd.DataFrame(rebalance_weights, index=rebalance_dates)
        else:
            rebalance_history = pd.DataFrame(columns=self.asset_universe)
        
        return {
            'full_history': full_history,
            'rebalance_history': rebalance_history
        }


def main():
    """Main function for testing Strategy 5."""
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Asset universe
    asset_universe = ["Nifty_10_Year", "Nifty_5_year", "NIFTY_500", "Gold", "Silver"]
    
    # Initialize strategy
    strategy = CorrelationAwareMomentumRiskParityStrategy(asset_universe)
    
    # Load test data
    data_path = os.path.join(os.path.dirname(__file__), '../input_data/Combined_input.xlsx')
    try:
        historical_data = pd.read_excel(data_path, parse_dates=['Date'])
        historical_data.set_index('Date', inplace=True)
        logger.info(f"Loaded {len(historical_data)} days of data")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Test strategy
    min_days = 252
    if len(historical_data) < min_days:
        logger.error(f"Not enough data (need {min_days}, have {len(historical_data)})")
        return
    
    # Get test date (last available date)
    test_date = historical_data.index[-1]
    logger.info(f"Testing strategy on {test_date}")
    
    # Calculate weights
    weights = strategy.calculate_target_weights(test_date, historical_data)
    
    # Display results
    print("\n" + "="*60)
    print("STRATEGY 5: CORRELATION-AWARE MOMENTUM + RISK PARITY")
    print("="*60)
    
    info = strategy.get_strategy_info()
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\nTarget Weights for {test_date.strftime('%Y-%m-%d')}:")
    print("-" * 40)
    total_weight = 0
    for asset, weight in weights.items():
        if weight > 0.001:
            print(f"{asset:15} {weight:8.2%}")
            total_weight += weight
    
    print(f"{'Total':15} {total_weight:8.2%}")
    
    # Test weight history
    print(f"\nGenerating weight history...")
    history = strategy.get_weight_history(historical_data, min_days=min_days)
    
    print(f"Full history: {len(history['full_history'])} days")
    print(f"Rebalance history: {len(history['rebalance_history'])} rebalance dates")
    
    if not history['rebalance_history'].empty:
        print(f"\nRecent rebalance weights:")
        print(history['rebalance_history'].tail(3).round(4))


if __name__ == "__main__":
    main() 