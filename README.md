# ReSolve Adaptive Asset Allocation - Python Implementation

A comprehensive Python implementation of the ReSolve Adaptive Asset Allocation methodology specifically adapted for Indian markets. This implementation follows the research paper "ReSolve Adaptive Asset Allocation: A Primer" and provides a complete backtesting and analysis framework.

## 🎯 Overview

This implementation provides:

- **Complete ReSolve Methodology**: Multi-timeframe momentum calculation, risk-weighted optimization, and volatility targeting
- **Indian Market Focus**: Asset universe covering Indian equities, bonds, REITs, and global diversification
- **Professional Architecture**: Clean, modular design with proper abstractions and interfaces
- **Comprehensive Backtesting**: Full backtesting engine with transaction costs and performance analysis
- **Production Ready**: Extensible design for Bloomberg Terminal integration

## 📁 Project Structure

```
resolve_python_implementation/
├── src/
│   ├── core/
│   │   ├── base.py              # Base classes and interfaces
│   │   ├── momentum.py          # Momentum calculation implementations
│   │   └── strategy.py          # Main strategy engine
│   ├── data/
│   │   └── provider.py          # Data provider with synthetic Indian market data
│   ├── risk/
│   │   └── calculator.py        # Risk metrics and volatility estimation
│   ├── optimization/
│   │   └── optimizer.py         # Portfolio optimization implementations
│   └── analytics/
│       └── backtest.py          # Backtesting engine and performance analysis
├── config/                      # Configuration files
├── data/                        # Data storage (raw and processed)
├── results/                     # Backtest results and reports
├── tests/                       # Unit tests
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

1. **Clone or extract the implementation**:
   ```bash
   cd resolve_python_implementation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demonstration**:
   ```bash
   python main.py
   ```

### Basic Usage

```python
from src.core.base import StrategyConfiguration, AssetUniverse, Asset
from src.data.provider import IndianMarketDataProvider
from src.core.strategy import ResolveStrategyEngine
from src.analytics.backtest import ResolveBacktestEngine
from datetime import datetime, timedelta

# Create asset universe
assets = [
    Asset("NIFTY50", "Nifty 50 Index", "equity", "India"),
    Asset("NIFTY10YRGOVT", "Nifty 10 Year Government Bond Index", "fixed_income", "India"),
    # ... more assets
]
asset_universe = AssetUniverse(assets)

# Configure strategy
config = StrategyConfiguration(
    target_volatility=0.10,
    momentum_lookback_periods=[1, 3, 6, 9, 12],
    rebalancing_frequency='monthly'
)

# Initialize components
data_provider = IndianMarketDataProvider()
strategy_engine = ResolveStrategyEngine(data_provider, asset_universe, config)

# Calculate current allocation
current_weights = strategy_engine.calculate_target_weights(datetime.now())

# Run backtest
backtest_engine = ResolveBacktestEngine(strategy_engine)
results = backtest_engine.run_backtest(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

## 🏗️ Architecture

### Core Components

1. **Strategy Engine** (`src/core/strategy.py`)
   - Main orchestrator implementing the complete ReSolve methodology
   - Coordinates momentum calculation, risk estimation, and portfolio optimization
   - Handles rebalancing logic and volatility targeting

2. **Momentum Calculator** (`src/core/momentum.py`)
   - Time-series momentum calculation
   - Cross-sectional momentum ranking
   - Combined momentum approach with configurable weighting

3. **Risk Calculator** (`src/risk/calculator.py`)
   - Volatility estimation (simple, EWM, GARCH)
   - Correlation matrix calculation
   - Comprehensive risk metrics (VaR, Sharpe, Sortino, etc.)

4. **Portfolio Optimizer** (`src/optimization/optimizer.py`)
   - Mean-variance optimization with momentum signals
   - Risk parity optimization
   - Volatility targeting and leverage adjustment

5. **Data Provider** (`src/data/provider.py`)
   - Synthetic Indian market data generation
   - Extensible interface for Bloomberg Terminal integration
   - Realistic price dynamics and correlations

6. **Backtesting Engine** (`src/analytics/backtest.py`)
   - Historical simulation with look-ahead bias prevention
   - Transaction cost modeling
   - Comprehensive performance analysis and reporting

### Design Principles

- **Interface-Based Design**: All major components implement interfaces for easy testing and extension
- **Separation of Concerns**: Clear separation between data, calculation, optimization, and analysis
- **Configurability**: Extensive configuration options without code changes
- **Extensibility**: Easy to add new assets, optimization methods, or data sources
- **Production Ready**: Proper error handling, logging, and validation

## 📊 Asset Universe

The implementation includes a comprehensive Indian market asset universe:

### Indian Equities
- **NIFTY50**: Nifty 50 Index
- **NIFTYNEXT50**: Nifty Next 50 Index
- **NIFTYMIDCAP**: Nifty Midcap 150 Index
- **NIFTYSMALLCAP**: Nifty Smallcap 250 Index

### Indian Fixed Income
- **NIFTY10YRGOVT**: Nifty 10 Year Government Bond Index
- **NIFTYCORPBOND**: Nifty Corporate Bond Index
- **NIFTYGSEC**: Nifty Government Securities Index

### Indian Real Estate
- **NIFTYREIT**: Nifty REIT Index
- **NIFTYREALTY**: Nifty Realty Index

### Global Diversification
- **MSCIWORLD**: MSCI World Index (INR hedged)
- **MSCIEM**: MSCI Emerging Markets Index (INR hedged)
- **GOLD**: Gold (INR)
- **CRUDE**: Crude Oil (INR)

### Currency
- **USDINR**: USD/INR Exchange Rate

## ⚙️ Configuration

### Strategy Parameters

```python
StrategyConfiguration(
    target_volatility=0.10,              # 10% target volatility
    momentum_lookback_periods=[1,3,6,9,12], # Momentum periods in months
    rebalancing_frequency='monthly',      # Rebalancing frequency
    minimum_weight=0.0,                   # Minimum asset weight
    maximum_weight=0.25,                  # Maximum asset weight (25%)
    transaction_cost=0.001                # Transaction cost (0.1%)
)
```

### Customizable Components

- **Momentum Calculation**: Time-series, cross-sectional, or combined
- **Risk Estimation**: Simple, exponentially weighted, or GARCH volatility
- **Optimization Method**: Mean-variance, risk parity, or momentum-weighted
- **Rebalancing Frequency**: Daily, weekly, monthly, quarterly, or annual

## 📈 Performance Analysis

The backtesting engine provides comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Risk Metrics
- Maximum Drawdown
- Value at Risk (95%, 99%)
- Win Rate
- Tracking Error
- Information Ratio

### Transaction Analysis
- Total Transaction Costs
- Number of Rebalances
- Turnover Analysis

## 🔌 Bloomberg Integration

For production use with real data, extend the `IDataProvider` interface:

```python
class BloombergDataProvider(IDataProvider):
    def __init__(self, session):
        self.session = session  # Bloomberg API session
    
    def get_price_data(self, symbol, start_date, end_date):
        # Implement Bloomberg API calls
        # Required fields: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME
        pass
```

### Required Bloomberg Tickers

Map the asset symbols to Bloomberg tickers:
- `NIFTY50` → `NIFTY Index`
- `NIFTY10YRGOVT` → `NIFTY10YR Index`
- `GOLD` → `GOLDS Comdty` (INR adjusted)
- etc.

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## 📝 Example Output

```
ReSolve Adaptive Asset Allocation - Indian Markets Implementation
================================================================================

Asset Universe: 14 assets
Target Volatility: 10.0%
Momentum Periods: [1, 3, 6, 9, 12] months

CURRENT TARGET ALLOCATION:
--------------------------------------------------
NIFTY50         Nifty 50 Index                    18.45%
GOLD            Gold (INR)                        15.23%
NIFTY10YRGOVT   Nifty 10 Year Government Bond     12.67%
MSCIWORLD       MSCI World Index (INR)            11.89%
...

BACKTEST RESULTS:
------------------------------
Total Return: 12.45%
Annualized Return: 11.23%
Volatility: 9.87%
Sharpe Ratio: 0.847
Maximum Drawdown: -8.23%
```

## 🔧 Customization

### Adding New Assets

```python
new_asset = Asset(
    symbol="NEWASSET",
    name="New Asset Name",
    asset_type="equity",  # equity, fixed_income, commodity, etc.
    region="India"
)
asset_universe.add_asset(new_asset)
```

### Custom Optimization

```python
class CustomOptimizer(IPortfolioOptimizer):
    def optimize_weights(self, expected_returns, covariance_matrix, constraints):
        # Implement custom optimization logic
        pass
```

### Custom Risk Models

```python
class CustomRiskCalculator(IRiskCalculator):
    def calculate_volatility(self, returns, window):
        # Implement custom volatility estimation
        pass
```

## 📚 Research Background

This implementation is based on the ReSolve Asset Management research paper:
**"ReSolve Adaptive Asset Allocation: A Primer"**

### Key Methodology Components

1. **Multi-Timeframe Momentum**: Uses 1, 3, 6, 9, and 12-month lookback periods
2. **Risk-Weighted Optimization**: Combines momentum signals with mean-variance optimization
3. **Volatility Targeting**: Dynamically adjusts leverage to maintain target volatility
4. **Transaction Cost Awareness**: Incorporates realistic transaction costs in optimization

### Adaptations for Indian Markets

- **Asset Universe**: Focused on Indian equity indices, bonds, and REITs
- **Risk-Free Rate**: Uses 6.5% (typical Indian government bond yield)
- **Currency Hedging**: Global assets are INR-hedged
- **Market Characteristics**: Accounts for Indian market volatility and correlation patterns

## 🤝 Contributing

To extend or modify the implementation:

1. Follow the interface-based design patterns
2. Add comprehensive unit tests for new components
3. Update documentation for new features
4. Ensure backward compatibility

## 📄 License

This implementation is provided for educational and research purposes. Please ensure compliance with any applicable licenses for the underlying research methodology.

## 🆘 Support

For questions or issues:

1. Check the comprehensive logging output in `resolve_strategy.log`
2. Review the generated backtest reports for detailed analysis
3. Examine the CSV output files for raw data

## 🔮 Future Enhancements

Potential areas for extension:

- **Machine Learning Integration**: ML-based momentum signals
- **Alternative Risk Models**: Factor models, regime-aware risk estimation
- **Multi-Currency Support**: Dynamic currency hedging
- **Real-Time Execution**: Live trading integration
- **Advanced Analytics**: Attribution analysis, scenario testing

---

**Note**: This implementation uses synthetic data for demonstration. For production use, integrate with Bloomberg Terminal or other professional data providers for accurate historical and real-time market data.

