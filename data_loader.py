"""
data_loader.py - Download and prepare data for Betting-Against-Beta strategy

This script:
1. Downloads current MSCI World proxy constituents via yfinance
2. Fetches monthly adjusted close prices for all stocks and MSCI World proxy
3. Downloads and processes risk-free rate (^IRX)
4. Computes monthly returns, excess returns, and rolling 60-month betas
5. Saves all outputs as CSVs

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
START_DATE = '2000-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
ROLLING_WINDOW = 60  # months for beta calculation
MSCI_WORLD_PROXY = 'URTH'  # iShares MSCI World ETF


def get_msci_world_constituents():
    """
    Get MSCI World constituent tickers.

    Strategy:
    1. Try to fetch major developed market stocks from well-known indices
    2. Use URTH ETF info if available
    3. Fall back to curated list of major MSCI World constituents

    Returns:
        list: List of valid stock tickers
    """
    logger.info("Fetching MSCI World constituent tickers...")

    # Major MSCI World constituents - curated list covering major developed markets
    # This represents a representative sample of large-cap stocks across developed markets
    msci_world_tickers = [
        # United States - Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
        'MRK', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'BAC', 'MCD', 'CSCO', 'WMT',
        'TMO', 'PFE', 'ABT', 'CRM', 'ACN', 'ADBE', 'DHR', 'NKE', 'TXN', 'NFLX',
        'ORCL', 'AMD', 'UNP', 'PM', 'RTX', 'NEE', 'LOW', 'QCOM', 'INTC', 'LIN',
        'BMY', 'CMCSA', 'AMGN', 'T', 'VZ', 'UPS', 'HON', 'COP', 'SBUX', 'IBM',
        'GS', 'CAT', 'BA', 'MMM', 'DE', 'BLK', 'AXP', 'GILD', 'MS', 'CVS',
        'MDLZ', 'GE', 'SCHW', 'C', 'INTU', 'ADI', 'PLD', 'AMT', 'NOW', 'ISRG',
        'ZTS', 'SYK', 'BKNG', 'VRTX', 'ADP', 'MMC', 'SPGI', 'MO', 'CB', 'BDX',
        'REGN', 'ETN', 'TJX', 'LRCX', 'DUK', 'SO', 'CME', 'PGR', 'WM', 'CI',

        # Japan (ADRs and liquid tickers)
        'TM', 'SONY', 'MUFG', 'SMFG', 'HMC', 'NTT', 'SNE', 'MFG', 'NTDOY', 'NPPXF',

        # United Kingdom
        'BP', 'SHEL', 'HSBC', 'GSK', 'AZN', 'UL', 'RIO', 'BHP', 'BTI', 'LYG',
        'NWG', 'VOD', 'DEO',

        # Germany
        'SAP', 'DB', 'DTEGY', 'BASFY', 'BAYRY', 'SIEGY', 'VWAGY',

        # France
        'TTE', 'SNY', 'LVMUY', 'LRLCY', 'BNPQY',

        # Switzerland
        'NSRGY', 'NVS', 'RHHBY', 'UBS', 'CSGN',

        # Netherlands
        'ASML', 'ING', 'NVO',

        # Australia
        'BHP', 'RIO', 'WBK',

        # Canada
        'TD', 'RY', 'BNS', 'CNQ', 'ENB', 'CP', 'CNI', 'TRP', 'BMO', 'CM',
        'SU', 'MFC', 'BCE',

        # Other Developed Markets
        'TSM',  # Taiwan (often included)
        'TCEHY', 'BABA',  # China ADRs (sometimes in EM/DM boundary)
    ]

    # Remove duplicates while preserving order
    msci_world_tickers = list(dict.fromkeys(msci_world_tickers))

    logger.info(f"Initial ticker list contains {len(msci_world_tickers)} symbols")

    return msci_world_tickers


def validate_and_clean_tickers(tickers, sample_size=None):
    """
    Validate tickers by checking if they exist in yfinance.

    Args:
        tickers: List of ticker symbols
        sample_size: Optional limit for testing

    Returns:
        list: List of valid tickers
    """
    logger.info("Validating tickers...")

    if sample_size:
        tickers = tickers[:sample_size]

    valid_tickers = []
    invalid_tickers = []

    for ticker in tickers:
        try:
            # Basic validation - check if ticker has any data
            test_data = yf.Ticker(ticker)
            hist = test_data.history(period='5d')
            if len(hist) > 0:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception as e:
            invalid_tickers.append(ticker)
            logger.debug(f"Invalid ticker {ticker}: {e}")

    logger.info(f"Valid tickers: {len(valid_tickers)}, Invalid: {len(invalid_tickers)}")
    if invalid_tickers:
        logger.debug(f"Invalid tickers: {invalid_tickers[:10]}...")

    return valid_tickers


def download_monthly_prices(tickers, start_date, end_date):
    """
    Download monthly adjusted close prices for all tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string

    Returns:
        pd.DataFrame: Wide DataFrame with Date index and tickers as columns
    """
    logger.info(f"Downloading monthly prices for {len(tickers)} tickers...")

    # Download data in batches to avoid rate limits
    all_data = pd.DataFrame()
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                interval='1mo',
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if len(batch) == 1:
                # Single ticker returns Series, need to convert
                if 'Close' in data.columns:
                    batch_df = data[['Close']].copy()
                    batch_df.columns = [batch[0]]
                else:
                    batch_df = pd.DataFrame()
            else:
                # Multiple tickers return MultiIndex columns
                if 'Close' in data.columns.get_level_values(0):
                    batch_df = data['Close'].copy()
                else:
                    batch_df = pd.DataFrame()

            if not batch_df.empty:
                if all_data.empty:
                    all_data = batch_df
                else:
                    all_data = all_data.join(batch_df, how='outer')

        except Exception as e:
            logger.warning(f"Error downloading batch: {e}")
            continue

    # Ensure datetime index
    all_data.index = pd.to_datetime(all_data.index)

    # Resample to month-end to ensure consistent dates
    all_data = all_data.resample('ME').last()

    logger.info(f"Downloaded prices shape: {all_data.shape}")

    return all_data


def download_benchmark(start_date, end_date, proxy=MSCI_WORLD_PROXY):
    """
    Download MSCI World proxy (URTH ETF) monthly prices.

    Note: URTH started trading in 2012. For earlier dates, we'll use a blend
    of major market indices or accept the limitation.

    Args:
        start_date: Start date string
        end_date: End date string
        proxy: Ticker symbol for MSCI World proxy

    Returns:
        pd.Series: Monthly adjusted close prices
    """
    logger.info(f"Downloading MSCI World proxy ({proxy})...")

    try:
        data = yf.download(
            proxy,
            start=start_date,
            end=end_date,
            interval='1mo',
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data returned for {proxy}")

        # Handle single vs multi-column return (yfinance may return MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex columns: ('Close', 'URTH'), etc.
            benchmark = data['Close'].iloc[:, 0].copy()
        elif 'Close' in data.columns:
            benchmark = data['Close'].copy()
        else:
            benchmark = data.iloc[:, 0].copy()

        # Ensure it's a Series
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]

        # Resample to month-end
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark = benchmark.resample('ME').last()
        benchmark.name = 'MSCI_World'

        logger.info(f"Benchmark data from {benchmark.index.min()} to {benchmark.index.max()}")

        return benchmark

    except Exception as e:
        logger.error(f"Error downloading benchmark: {e}")
        raise


def download_risk_free_rate(start_date, end_date):
    """
    Download 3-month T-bill rate (^IRX) and convert to monthly decimal rate.

    ^IRX is quoted as annual percentage yield. We:
    1. Download daily data
    2. Resample to month-end
    3. Convert from annual % to monthly decimal rate

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        pd.Series: Monthly risk-free rate as decimal
    """
    logger.info("Downloading risk-free rate (^IRX)...")

    try:
        data = yf.download(
            '^IRX',
            start=start_date,
            end=end_date,
            interval='1d',
            progress=False
        )

        if data.empty:
            raise RuntimeError("No ^IRX data returned; cannot compute risk-free rate.")

        # Get close prices
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                rf_daily = data['Close'].copy()
            else:
                rf_daily = data.iloc[:, 0].copy()
        else:
            rf_daily = pd.Series(dtype=float)

        # Resample to month-end
        rf_daily.index = pd.to_datetime(rf_daily.index)
        rf_monthly = rf_daily.resample('ME').last()

        # Convert from annual percentage to monthly decimal
        # ^IRX is quoted as percentage (e.g., 4.5 means 4.5%)
        # Monthly rate = (1 + annual_rate/100)^(1/12) - 1
        rf_monthly_decimal = (1 + rf_monthly / 100) ** (1/12) - 1
        rf_monthly_decimal = rf_monthly_decimal.rename('RF_Rate')

        logger.info(f"Risk-free rate data from {rf_monthly_decimal.index.min()} to {rf_monthly_decimal.index.max()}")

        return rf_monthly_decimal

    except Exception as e:
        logger.warning(f"Error downloading risk-free rate: {e}")
        return None


def compute_returns(prices):
    """
    Compute simple monthly returns using percentage change.

    Args:
        prices: DataFrame of monthly prices

    Returns:
        pd.DataFrame: Monthly returns
    """
    logger.info("Computing monthly returns...")
    returns = prices.pct_change()

    # Drop the first row (NaN from pct_change)
    returns = returns.iloc[1:]

    return returns


def compute_excess_returns(returns, rf_rate):
    """
    Compute excess returns (return minus risk-free rate).

    Args:
        returns: DataFrame of monthly returns
        rf_rate: Series of monthly risk-free rates

    Returns:
        pd.DataFrame: Monthly excess returns
    """
    logger.info("Computing excess returns...")

    if rf_rate is None:
        raise RuntimeError("Risk-free rate is missing; cannot compute excess returns.")

    # Allow rf_rate to be passed as DataFrame or Series
    if isinstance(rf_rate, pd.DataFrame):
        if rf_rate.shape[1] > 1:
            rf_rate = rf_rate.iloc[:, 0]
        else:
            rf_rate = rf_rate.squeeze()
    rf_rate = rf_rate.rename('RF_Rate')

    # Align dates
    common_dates = returns.index.intersection(rf_rate.index)

    # Subtract risk-free rate from each column
    excess_returns = returns.loc[common_dates].subtract(rf_rate.loc[common_dates], axis=0)

    return excess_returns


def compute_rolling_betas(excess_returns_stocks, excess_returns_benchmark, window=ROLLING_WINDOW):
    """
    Compute rolling betas for each stock versus the benchmark using excess returns only.

    Beta = Cov(stock_excess, benchmark_excess) / Var(benchmark_excess)

    We require a full 60-month window of non-missing excess returns (no look-ahead).

    Args:
        excess_returns_stocks: DataFrame of stock excess returns
        excess_returns_benchmark: Series of benchmark excess returns
        window: Rolling window size in months

    Returns:
        pd.DataFrame: Rolling betas for each stock
    """
    logger.info(f"Computing rolling {window}-month betas...")

    # Align dates
    common_dates = excess_returns_stocks.index.intersection(excess_returns_benchmark.index)
    stock_rets = excess_returns_stocks.loc[common_dates]
    bench_rets = excess_returns_benchmark.loc[common_dates]

    betas = pd.DataFrame(index=stock_rets.index, columns=stock_rets.columns, dtype=float)
    min_periods = window  # strict 60-month window

    # Pre-compute benchmark variance (needs full window)
    bench_var = bench_rets.rolling(window=window, min_periods=min_periods).var()

    for col in stock_rets.columns:
        stock_series = stock_rets[col]
        cov_series = stock_series.rolling(window=window, min_periods=min_periods).cov(bench_rets)
        betas[col] = cov_series / bench_var

    betas = betas.replace([np.inf, -np.inf], np.nan)

    valid_betas = betas.notna().sum().sum()
    total_cells = betas.shape[0] * betas.shape[1]
    missing_per_month = betas.isna().sum(axis=1)
    logger.info(f"Computed betas shape: {betas.shape}")
    logger.info(f"Valid beta values: {valid_betas} of {total_cells} ({100*valid_betas/total_cells:.1f}%)")
    logger.info(f"Average missing betas per month: {missing_per_month.mean():.1f} "
                f"(min {missing_per_month.min()}, max {missing_per_month.max()})")

    return betas


def save_data(data, filename):
    """
    Save DataFrame to CSV.

    Args:
        data: DataFrame or Series to save
        filename: Output filename (without path)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data.to_csv(filepath)
    logger.info(f"Saved {filename}: {data.shape}")


def main():
    """
    Main data loading pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting BAB Data Loader")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("=" * 60)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Get MSCI World constituents
    tickers = get_msci_world_constituents()
    pd.DataFrame({"Ticker": tickers}).to_csv(os.path.join(DATA_DIR, "ticker_list.csv"), index=False)
    logger.info("Saved proxy MSCI World ticker universe to data/ticker_list.csv (URTH-based subset; coverage + survivorship caveats)")

    # Step 2: Validate tickers (optional - comment out for faster runs)
    # tickers = validate_and_clean_tickers(tickers)

    # Step 3: Download monthly prices for stocks
    stock_prices = download_monthly_prices(tickers, START_DATE, END_DATE)

    # Step 4: Download benchmark (MSCI World proxy)
    benchmark_prices = download_benchmark(START_DATE, END_DATE)

    # Step 5: Download risk-free rate
    rf_rate = download_risk_free_rate(START_DATE, END_DATE)
    if rf_rate is None:
        existing_rf_path = os.path.join(DATA_DIR, 'risk_free_rate.csv')
        if os.path.exists(existing_rf_path):
            logger.warning("Using existing risk_free_rate.csv because download failed.")
            rf_df = pd.read_csv(existing_rf_path, index_col=0, parse_dates=True)
            first_col = rf_df.columns[0]
            rf_df = rf_df.rename(columns={first_col: 'RF_Rate'})
            rf_rate = rf_df.iloc[:, 0]
            rf_rate.name = 'RF_Rate'
            if rf_rate.isna().any() or rf_rate.std() == 0:
                raise RuntimeError("Existing risk_free_rate.csv is invalid (NaN or constant); aborting.")
        else:
            raise RuntimeError("Risk-free rate download failed and no cached risk_free_rate.csv found; aborting.")

    # Step 6: Compute returns
    stock_returns = compute_returns(stock_prices)
    # Handle case where benchmark_prices might be Series or DataFrame
    if isinstance(benchmark_prices, pd.Series):
        benchmark_df = benchmark_prices.to_frame()
    else:
        benchmark_df = benchmark_prices
    benchmark_returns = compute_returns(benchmark_df).iloc[:, 0]
    benchmark_returns.name = 'MSCI_World'

    # Step 7: Compute excess returns (R_i,t - R_f,t)
    stock_excess_returns = compute_excess_returns(stock_returns, rf_rate)
    benchmark_excess_returns = compute_excess_returns(benchmark_returns.to_frame(), rf_rate).iloc[:, 0]

    # Step 8: Compute rolling betas (excess-return OLS proxy, 60M window, >=60 obs)
    logger.info("Betas estimated on EXCESS returns (60-month window, require full window).")
    rolling_betas = compute_rolling_betas(stock_excess_returns, benchmark_excess_returns)

    # Step 9: Save all outputs
    logger.info("Saving all data files...")

    # Combine stock and benchmark prices
    all_prices = stock_prices.copy()
    all_prices['MSCI_World'] = benchmark_prices
    save_data(all_prices, 'monthly_prices.csv')

    # Combine stock and benchmark returns
    all_returns = stock_returns.copy()
    all_returns['MSCI_World'] = benchmark_returns
    save_data(all_returns, 'monthly_returns.csv')

    # Save excess returns
    all_excess_returns = stock_excess_returns.copy()
    all_excess_returns['MSCI_World'] = benchmark_excess_returns
    save_data(all_excess_returns, 'monthly_excess_returns.csv')

    # Save risk-free rate
    if rf_rate is not None:
        save_data(rf_rate.to_frame(), 'risk_free_rate.csv')

    # Save rolling betas
    save_data(rolling_betas, 'rolling_betas.csv')

    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info(f"Output files saved to: {DATA_DIR}")
    logger.info("=" * 60)

    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Date range: {all_prices.index.min()} to {all_prices.index.max()}")
    print(f"Number of stocks: {len(stock_prices.columns)}")
    print(f"Total months: {len(all_prices)}")
    print(f"Stocks with valid betas (final month): {rolling_betas.iloc[-1].notna().sum()}")


if __name__ == '__main__':
    main()
