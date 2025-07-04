import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import dask
from dask.distributed import Client
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
# Specifically suppress pandas warnings
pd.options.mode.chained_assignment = None


def backtester_quantity_based(
    returns_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    date_list: List[pd.Timestamp],
    transaction_cost_rate: float = 0.001,
    base_value: float = 1000,
    disable_tqdm: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest a portfolio using quantity-based rebalancing, supporting both long-only and long-short.
    Uses tqdm for progress bar in the main rebalance loop. Set disable_tqdm=True to turn off.
    Args:
        returns_df (pd.DataFrame): DataFrame with ['date', 'Accord Code', 'adj_close' or 'price'].
        portfolio_df (pd.DataFrame): DataFrame with ['Rebalance_Date', 'Accord Code', 'weights'].
        date_list (list): List of rebalance dates.
        transaction_cost_rate (float): Transaction cost rate per trade.
        base_value (float): Starting portfolio value.
        disable_tqdm (bool): If True, disables tqdm progress bar.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (index values, transaction costs)
    """
    returns_df = returns_df.copy()
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    portfolio_df['Rebalance_Date'] = pd.to_datetime(portfolio_df['Rebalance_Date'])
    date_list = pd.to_datetime(date_list)
    index_values = pd.DataFrame()
    transaction_costs_records = []
    current_quantities = {}
    for i in tqdm(range(len(date_list) - 1), dynamic_ncols=True, leave=False, disable=disable_tqdm):
        period = date_list[i]
        next_period = date_list[i + 1]
        temp_df = portfolio_df[portfolio_df['Rebalance_Date'] == period][['Accord Code', 'weights']].copy()
        if temp_df.empty:
            continue
        is_long_only = (temp_df['weights'] >= 0).all()
        codes = temp_df['Accord Code'].unique().tolist()
        price_data = returns_df[
            (returns_df['Accord Code'].isin(codes)) &
            (returns_df['date'] >= period) &
            (returns_df['date'] <= next_period)
        ].copy()
        if 'price' not in price_data.columns:
            if 'adj_close' in price_data.columns:
                price_data.rename(columns={'adj_close': 'price'}, inplace=True)
            else:
                raise KeyError("Missing 'price' or 'adj_close' column in returns_df")
        rebalance_prices = price_data.groupby('Accord Code').first().reset_index()
        rebalance_prices_dict = dict(zip(rebalance_prices['Accord Code'], rebalance_prices['price']))
        if is_long_only:
            target_quantities = {}
            for _, row in temp_df.iterrows():
                code = row['Accord Code']
                weight = row['weights']
                price = rebalance_prices_dict.get(code, None)
                if price and price > 0:
                    target_quantities[code] = (base_value * weight) / price
        else:
            long_df = temp_df[temp_df['weights'] >= 0].copy()
            short_df = temp_df[temp_df['weights'] < 0].copy()
            target_quantities = {}
            for _, row in pd.concat([long_df, short_df]).iterrows():
                code = row['Accord Code']
                weight = row['weights']
                price = rebalance_prices_dict.get(code, None)
                if price and price > 0:
                    leg_value = base_value  # allocate full to both legs
                    target_quantities[code] = (leg_value * weight) / price
        # Transaction cost
        transaction_cost = 0
        all_codes = set(current_quantities.keys()).union(set(target_quantities.keys()))
        for code in all_codes:
            prev_q = current_quantities.get(code, 0)
            new_q = target_quantities.get(code, 0)
            delta = abs(new_q - prev_q)
            price = rebalance_prices_dict.get(code, 0)
            transaction_cost += delta * price * transaction_cost_rate
        transaction_costs_records.append({
            'Date': period,
            'Transaction_Cost': transaction_cost,
            'Portfolio_Value_Before_Costs': base_value
        })
        base_value = base_value - transaction_cost
        current_quantities = target_quantities.copy()
        # Get prices over the holding period
        returns_period = returns_df[
            (returns_df['Accord Code'].isin(current_quantities.keys())) &
            (returns_df['date'] >= period) &
            (returns_df['date'] <= next_period)
        ].copy()
        returns_period['quantity'] = returns_period['Accord Code'].map(current_quantities)
        returns_period['quantity'] = returns_period['quantity'].fillna(0)
        if 'price' not in returns_period.columns:
            if 'adj_close' in returns_period.columns:
                returns_period.rename(columns={'adj_close': 'price'}, inplace=True)
            else:
                raise KeyError("Missing 'price' or 'adj_close' column during value computation")
        returns_period['position_value'] = returns_period['quantity'] * returns_period['price']
        if is_long_only:
            portfolio_values = returns_period.groupby('date')['position_value'].sum().reset_index()
            portfolio_values.columns = ['Date', 'portfolio_value']
            portfolio_values['index_levels'] = portfolio_values['portfolio_value'] / portfolio_values['portfolio_value'].iloc[0] * base_value
            base_value = portfolio_values['index_levels'].iloc[-1]
        else:
            returns_period['side'] = returns_period['quantity'].apply(lambda x: 'long' if x >= 0 else 'short')
            long_values = returns_period[returns_period['side'] == 'long'].groupby('date')['position_value'].sum()
            short_values = returns_period[returns_period['side'] == 'short'].groupby('date')['position_value'].sum()
            all_dates = sorted(list(returns_period['date'].unique()))
            long_values = long_values.reindex(all_dates, method='ffill').fillna(method='bfill').fillna(0)
            short_values = short_values.reindex(all_dates, method='ffill').fillna(method='bfill').fillna(0)
            long_pnl = long_values - long_values.iloc[0]
            short_pnl = short_values - short_values.iloc[0]
            net_pnl = long_pnl + short_pnl
            net_portfolio_value = base_value + net_pnl
            portfolio_values = pd.DataFrame({
                'Date': all_dates,
                'portfolio_value': net_portfolio_value,
            })
            portfolio_values['index_levels'] = portfolio_values['portfolio_value'] / portfolio_values['portfolio_value'].iloc[0] * base_value
            base_value = portfolio_values['index_levels'].iloc[-1]
        # If index_values is empty, initialize it with the new values
        if index_values.empty:
            index_values = portfolio_values[['Date', 'index_levels']].copy()
        else:
            # Merge the new values with existing ones, keeping the latest values for duplicate dates
            index_values = pd.concat([
                index_values,
                portfolio_values[['Date', 'index_levels']]
            ]).drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
            
    transaction_costs_df = pd.DataFrame(transaction_costs_records).set_index('Date')

    if not index_values.empty:
        index_values.set_index('Date', inplace=True)
        index_values.sort_index(inplace=True)
    return index_values, transaction_costs_df


def run_single_backtest(
    returns_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    date_list: List[pd.Timestamp],
    transaction_cost_rate: float,
    base_value: float,
    strategy_name: str
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run a single backtest and return index and transactions."""
    index, transactions = backtester_quantity_based(
        returns_df,
        portfolio_df,
        date_list,
        transaction_cost_rate=transaction_cost_rate,
        base_value=base_value
    )
    return strategy_name, (index, transactions)

def load_portfolio_weights(file_path: str) -> pd.DataFrame:
    """
    Load portfolio weights from the combined portfolio CSV file.
    
    Args:
        file_path (str): Path to the combined portfolio CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Rebalance_Date', 'Accord Code', 'weights']
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime and rename it to Rebalance_Date
    df['Rebalance_Date'] = pd.to_datetime(df['date'])
    
    # Rename accord_code to Accord Code
    df = df.rename(columns={'accord_code': 'Accord Code'})
    
    # Rename weight to weights
    df = df.rename(columns={'weight': 'weights'})
    
    # Select and reorder required columns
    df = df[['Rebalance_Date', 'Accord Code', 'weights']]
    
    return df

def get_rebalance_dates(portfolio_df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Extract unique rebalance dates from the portfolio DataFrame.
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with 'Rebalance_Date' column
        
    Returns:
        List[pd.Timestamp]: List of unique rebalance dates sorted in ascending order
    """
    # Get unique dates and sort them
    rebalance_dates = sorted(portfolio_df['Rebalance_Date'].unique())
    
    return rebalance_dates

def run_backtests(returns_df: pd.DataFrame, path: str):
    """Run backtests in parallel using dask"""
    print("Starting backtests...")
    print(f"Returns DataFrame shape: {returns_df.shape}")
    print(f"Returns DataFrame head:\n{returns_df.head()}")
    
    # Initialize dask client
    client = Client()
    
    # Define strategies to backtest
    strategies = {
        "TC0.1_correlation_aware_momentum_risk_parity": {
            "portfolio_file": r"D:\Arsh\resolve-implementation\Portfolios\correlation_aware_momentum_risk_parity_rebalance_history_long.csv ",
            "transaction_cost_rate": 0.001,
            "base_value": 1000
        },
        "TC0_correlation_aware_momentum_risk_parity": {
            "portfolio_file": r"D:\Arsh\resolve-implementation\Portfolios\correlation_aware_momentum_risk_parity_rebalance_history_long.csv ",
            "transaction_cost_rate": 0.000,
            "base_value": 1000
        }
    }
    
    # Create delayed tasks for each strategy
    delayed_tasks = []
    for strategy_name, config in strategies.items():
        print(f"\nProcessing strategy: {strategy_name}")
        portfolio_file = config["portfolio_file"]
        portfolio_df = load_portfolio_weights(portfolio_file)
        print(f"Portfolio DataFrame shape: {portfolio_df.shape}")
        print(f"Portfolio DataFrame head:\n{portfolio_df.head()}")
        
        date_list = get_rebalance_dates(portfolio_df)
        print(f"Number of rebalance dates: {len(date_list)}")
        print(f"First and last rebalance dates: {date_list[0]} to {date_list[-1]}")
        
        task = dask.delayed(run_single_backtest)(
            returns_df,
            portfolio_df,
            date_list,
            config["transaction_cost_rate"],
            config["base_value"],
            strategy_name
        )
        delayed_tasks.append(task)
    
    # Compute all backtests in parallel
    print("\nComputing backtests...")
    results = dask.compute(*delayed_tasks)
    print("Backtests completed!")
    
    # Process and plot results
    print("\nProcessing results...")
    results_df = pd.DataFrame()
    transaction_costs_df = pd.DataFrame()
    
    for strategy_name, (index, transactions) in results:
        print(f"\nProcessing results for strategy: {strategy_name}")
        print(f"Index shape: {index.shape}")
        print(f"Index head:\n{index.head()}")
        
        # Add strategy performance to results DataFrame
        index = index.reset_index()
        index.columns = ['date', 'index_levels']
        index['strategy'] = strategy_name
        
        transactions = transactions.reset_index()
        transactions['strategy'] = strategy_name
        
        results_df = pd.concat([results_df, index], ignore_index=True)
        transaction_costs_df = pd.concat([transaction_costs_df, transactions], ignore_index=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(path, 'final_result')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving results...")
    print(f"Final results shape: {results_df.shape}")
    print(f"Final results head:\n{results_df.head()}")
    
    # Save results to Excel
    results_df.to_excel(os.path.join(output_dir, 'strategy_performance_results.xlsx'), index=False)
    transaction_costs_df.to_excel(os.path.join(output_dir, 'transaction_costs_results.xlsx'), index=False)
    print("Results saved successfully!")
    
    # Close dask client
    client.close()

    

def load_price_data(file_path: str) -> pd.DataFrame:
    """
    Load price data from the combined portfolio CSV file.
    
    Args:
        file_path (str): Path to the combined portfolio CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'Accord Code', 'adj_close']
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Map the column names to match expected format
    df = df.rename(columns={
        'accord_code': 'Accord Code',
        'price': 'adj_close'
    })
    
    # Select and reorder required columns
    df = df[['date', 'Accord Code', 'adj_close']]
    # df['Accord Code'] = df['Accord Code'].astype(int)
    
    return df

def prepare_returns_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the returns DataFrame from price data, including daily return, adj_close pivot, and stock_returns.
    Args:
        price_df (pd.DataFrame): DataFrame with columns ['date', 'Accord Code', 'adj_close'].
    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'Accord Code', 'adj_close', 'daily return'].
    """
    df = price_df.copy()
    df = df.sort_values(by=['Accord Code', 'date'], ascending=True)
    df['daily return'] = df.groupby('Accord Code')['adj_close'].pct_change(1)
    df['daily return'].fillna(0, inplace=True)
    return df


if __name__ == "__main__":
    print("Starting main execution...")
    price_file = r"D:\Arsh\resolve-implementation\Portfolios\correlation_aware_momentum_risk_parity_rebalance_history_long.csv"
    print(f"\nLoading price data from: {price_file}")
    price_df = load_price_data(price_file)
    print(f"Price DataFrame shape: {price_df.shape}")
    print(f"Price DataFrame head:\n{price_df.head()}")
    
    print("\nPreparing returns DataFrame...")
    returns_df = prepare_returns_df(price_df)
    print(f"Returns DataFrame shape: {returns_df.shape}")
    print(f"Returns DataFrame head:\n{returns_df.head()}")
    
    run_backtests(
        returns_df=returns_df,
        path=r"D:\Arsh\resolve-implementation\Portfolios"
    )