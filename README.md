# Asset Allocation Strategies

This package implements four different asset allocation strategies for Indian markets, providing daily signals with monthly rebalancing and portfolio weight outputs.

## Asset Universe

The strategies are designed for the following Indian market assets:
- **NIFTY500**: Nifty 500 TRI (equity)
- **NIFTY10YRGOVT**: 10-year G-Sec index (long-term bonds)
- **NIFTY5YRGOVT**: 5-year G-Sec index (short-term bonds)
- **GOLD**: Gold (MCX spot)
- **SILVER**: Silver (MCX spot)
- **CASH**: Cash/Debt instruments

## Strategies

### 1. Equal-Weight Portfolio (Monthly Rebalanced)

**Concept**: Naïve baseline strategy that allocates equal capital weight to each asset class.

**Methodology**:
- Each asset gets weight `w_i = 1/N` (for N assets)
- Monthly rebalancing to restore equal weights
- Daily weight drift tracking between rebalances

**Key Features**:
- Simple and diversified approach
- No initial preference among asset types
- Typically results in ~40% debt allocation (5Y + 10Y bonds)

### 2. Volatility-Adjusted Risk Parity (60-Day Inverse-Vol Weighting)

**Concept**: Allocates weights inversely proportional to each asset's volatility, aiming for equal risk contribution.

**Methodology**:
- Uses 60-day rolling window for volatility calculation
- Weights calculated as `w_i = (1/σ_i) / Σ(1/σ_j)`
- Monthly rebalancing to incorporate new volatility data

**Key Features**:
- Balances risk across portfolio
- Overweights lower-volatility assets (bonds)
- Underweights higher-volatility assets (equities, commodities)

### 3. Momentum-Based Selection (6-Month Return Momentum Strategy)

**Concept**: Dynamically selects assets based on recent performance, holding trending assets.

**Methodology**:
- Calculates 6-month returns for all assets
- Selects assets with above-average momentum
- Equal weights among selected assets
- Non-selected assets allocated to safe debt

**Key Features**:
- Captures momentum effect
- Can rotate between asset classes
- Maintains significant debt allocation when momentum is weak

### 4. Momentum + Risk Parity Hybrid

**Concept**: Combines momentum selection with volatility weighting for optimal risk-adjusted returns.

**Methodology**:
- Step 1: Select assets with above-average 6-month momentum
- Step 2: Apply inverse-volatility weighting to selected assets
- Monthly rebalancing for both momentum and volatility updates

**Key Features**:
- Best of both worlds: momentum offense + risk parity defense
- Expected superior risk-adjusted performance
- Dynamic asset selection with risk control

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running All Strategies

```python
from asset_allocation.main import main

# Run comparison of all strategies
main()
```

### Running Single Strategy

```python
from asset_allocation.main import run_single_strategy

# Run individual strategy with user input
run_single_strategy()
```

### Direct Strategy Usage

```python
from asset_allocation.strategy1 import EqualWeightStrategy
from asset_allocation.strategy2 import VolatilityAdjustedStrategy
from asset_allocation.strategy3 import MomentumSelectionStrategy
from asset_allocation.strategy4 import MomentumRiskParityStrategy

# Initialize strategy
strategy = EqualWeightStrategy(asset_universe=['NIFTY500', 'NIFTY10YRGOVT', 'GOLD'])

# Calculate target weights
weights = strategy.calculate_target_weights(current_date, historical_data)

# Get daily signals
signals = strategy.get_daily_signals(current_date, historical_data)
```

## Output Format

Each strategy provides:
- **Daily Signals**: Portfolio weights for each trading day
- **Monthly Rebalancing**: Target weights updated monthly
- **Portfolio Weights**: Dictionary mapping asset symbols to weights (0.0 to 1.0)

Example output:
```python
{
    'NIFTY500': 0.20,      # 20% allocation
    'NIFTY10YRGOVT': 0.30, # 30% allocation
    'NIFTY5YRGOVT': 0.25,  # 25% allocation
    'GOLD': 0.15,          # 15% allocation
    'SILVER': 0.10,        # 10% allocation
    'CASH': 0.00           # 0% allocation
}
```

## Data Requirements

For production use, connect to Bloomberg Terminal with:

**Required Bloomberg Fields**:
- PX_LAST (Last Price)
- PX_OPEN (Open Price)
- PX_HIGH (High Price)
- PX_LOW (Low Price)
- PX_VOLUME (Volume)

**Required Tickers**:
- NIFTY500: Nifty 500 TRI
- NIFTY10YRGOVT: Nifty 10 Year Government Bond Index
- NIFTY5YRGOVT: Nifty 5 Year Government Bond Index
- GOLD: Gold MCX Spot
- SILVER: Silver MCX Spot

**Data Specifications**:
- Historical Data: Minimum 2 years of daily data
- Update Frequency: Daily (end of day)
- Currency: All prices in INR

## Key Features

- **Modular Design**: Each strategy is implemented as a separate class
- **Type Annotations**: Full type hints for better code quality
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Robust error handling with fallbacks
- **Documentation**: Google-style docstrings for all methods
- **Testing Ready**: Designed for easy unit testing

## Performance Considerations

- **Volatility Calculation**: Uses rolling windows for efficient computation
- **Memory Management**: Optimized for large historical datasets
- **Rebalancing Logic**: Efficient monthly rebalancing detection
- **Weight Drift Tracking**: Monitors portfolio drift between rebalances

## Extensions

The framework is designed for easy extension:
- Add new strategies by implementing the same interface
- Modify asset universe for different markets
- Adjust rebalancing frequencies
- Add correlation-based optimizations
- Implement transaction cost modeling

## License

This project is licensed under the MIT License. 