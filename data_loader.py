"""
data_loader.py - Download and prepare data for Betting-Against-Beta strategy

This script:
1. Downloads monthly adjusted close prices for MSCI World constituents
2. Downloads S&P 500 (^GSPC) as market benchmark
3. Downloads and processes risk-free rate (^IRX)
4. Computes monthly returns and excess returns
5. Computes rolling 60-month betas vs the benchmark
6. Saves all outputs as CSVs

Key Design:
- Uses curated list of major stocks from developed markets
- S&P 500 (^GSPC) as benchmark (full history from 1950s)
- 60-month rolling window for beta estimation
- Date range: 1995-2014 (first valid betas from 2000)

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import (
    START_DATE, END_DATE, ROLLING_WINDOW, MIN_PERIODS_BETA,
    DATA_DIR, BENCHMARK_TICKER, MSCI_WORLD_TICKERS,
    DOWNLOAD_BATCH_SIZE, ensure_directories,
    MIN_DATA_COVERAGE, REQUIRE_FULL_HISTORY,
    MSCI_WORLD_PROXIES, USE_COMPOSITE_MSCI
)


def download_monthly_prices(tickers, start_date, end_date):
    """
    Download monthly adjusted close prices for all tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        pd.DataFrame: Wide DataFrame with Date index and tickers as columns
    """
    logger.info(f"Downloading monthly prices for {len(tickers)} tickers...")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_data = pd.DataFrame()
    batch_size = DOWNLOAD_BATCH_SIZE
    failed_tickers = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) - 1) // batch_size + 1
        logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch)} tickers)")

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

            if data.empty:
                logger.warning(f"No data returned for batch {batch_num}")
                failed_tickers.extend(batch)
                continue

            # Handle single vs multiple tickers
            if len(batch) == 1:
                if 'Close' in data.columns:
                    batch_df = data[['Close']].copy()
                    batch_df.columns = [batch[0]]
                else:
                    batch_df = pd.DataFrame()
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0):
                        batch_df = data['Close'].copy()
                    else:
                        batch_df = pd.DataFrame()
                elif 'Close' in data.columns:
                    batch_df = data[['Close']].copy()
                else:
                    batch_df = pd.DataFrame()

            if not batch_df.empty:
                if all_data.empty:
                    all_data = batch_df
                else:
                    all_data = all_data.join(batch_df, how='outer')

        except Exception as e:
            logger.warning(f"Error downloading batch {batch_num}: {e}")
            failed_tickers.extend(batch)
            continue

    if all_data.empty:
        raise RuntimeError("No price data could be downloaded!")

    # Ensure datetime index
    all_data.index = pd.to_datetime(all_data.index)

    # Resample to month-end for consistent dates
    all_data = all_data.resample('ME').last()

    # Strict data filtering for continuous data
    logger.info("Applying strict data quality filters...")
    initial_count = len(all_data.columns)

    # Step 1: Filter for minimum data coverage (default 95%)
    coverage = all_data.notna().sum() / len(all_data)
    valid_coverage = coverage[coverage >= MIN_DATA_COVERAGE].index
    all_data = all_data[valid_coverage]
    logger.info(f"After {MIN_DATA_COVERAGE*100:.0f}% coverage filter: {len(all_data.columns)} tickers "
                f"(dropped {initial_count - len(all_data.columns)})")

    # Step 2: Require data from start date (if configured)
    if REQUIRE_FULL_HISTORY:
        # Check if first valid date is within 3 months of start date
        first_valid = all_data.apply(lambda x: x.first_valid_index())
        start_threshold = pd.Timestamp(START_DATE) + pd.DateOffset(months=3)
        valid_start = first_valid[first_valid <= start_threshold].index
        before_count = len(all_data.columns)
        all_data = all_data[valid_start]
        logger.info(f"After start date filter: {len(all_data.columns)} tickers "
                    f"(dropped {before_count - len(all_data.columns)})")

    # Step 3: Check for data gaps (no more than 2 consecutive missing months)
    def has_large_gaps(series, max_gap=2):
        is_na = series.isna()
        if not is_na.any():
            return False
        # Count consecutive NaNs
        gaps = is_na.astype(int).groupby((~is_na).cumsum()).sum()
        return gaps.max() > max_gap

    no_gaps = [col for col in all_data.columns if not has_large_gaps(all_data[col])]
    before_count = len(all_data.columns)
    all_data = all_data[no_gaps]
    logger.info(f"After gap filter: {len(all_data.columns)} tickers "
                f"(dropped {before_count - len(all_data.columns)})")

    # Forward fill any remaining small gaps
    all_data = all_data.ffill(limit=2)

    logger.info(f"Final dataset: {all_data.shape}")
    logger.info(f"Date range: {all_data.index.min()} to {all_data.index.max()}")

    if failed_tickers:
        logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

    return all_data


def download_benchmark(start_date, end_date, ticker=BENCHMARK_TICKER):
    """
    Download market benchmark monthly prices.

    Args:
        start_date: Start date string
        end_date: End date string
        ticker: Ticker symbol for benchmark (default: ^GSPC)

    Returns:
        pd.Series: Monthly adjusted close prices
    """
    logger.info(f"Downloading benchmark ({ticker})...")

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1mo',
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            benchmark = data['Close'].iloc[:, 0].copy()
        elif 'Close' in data.columns:
            benchmark = data['Close'].copy()
        else:
            benchmark = data.iloc[:, 0].copy()

        # Ensure Series
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]

        # Resample to month-end
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark = benchmark.resample('ME').last()
        benchmark.name = 'Benchmark'

        logger.info(f"Benchmark data: {len(benchmark)} months")
        logger.info(f"Date range: {benchmark.index.min()} to {benchmark.index.max()}")

        return benchmark

    except Exception as e:
        logger.error(f"Error downloading benchmark: {e}")
        raise


def download_msci_world_proxy(start_date, end_date):
    """
    Download and construct MSCI World proxy from weighted regional indices.

    Uses a market-cap weighted combination:
    - S&P 500 (USA): ~60%
    - STOXX Europe 600: ~25%
    - Nikkei 225 (Japan): ~10%
    - S&P/TSX (Canada): ~5%

    If composite construction fails, falls back to S&P 500.

    Returns:
        pd.Series: Monthly returns for MSCI World proxy
    """
    logger.info("Constructing MSCI World proxy from regional indices...")

    if not USE_COMPOSITE_MSCI:
        logger.info("Composite MSCI disabled, using S&P 500 as benchmark")
        prices = download_benchmark(start_date, end_date, BENCHMARK_TICKER)
        prices.name = 'MSCI_World'
        return prices

    index_returns = {}
    available_weight = 0.0

    for ticker, weight in MSCI_WORLD_PROXIES.items():
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1mo',
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                logger.warning(f"No data for {ticker}, skipping")
                continue

            # Extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'].iloc[:, 0].copy()
            elif 'Close' in data.columns:
                prices = data['Close'].copy()
            else:
                prices = data.iloc[:, 0].copy()

            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            # Resample to month-end
            prices.index = pd.to_datetime(prices.index)
            prices = prices.resample('ME').last()

            # Compute returns
            returns = prices.pct_change().dropna()

            if len(returns) > 0:
                index_returns[ticker] = returns
                available_weight += weight
                logger.info(f"  {ticker}: {len(returns)} months, weight={weight:.0%}")

        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
            continue

    if not index_returns:
        logger.warning("No regional indices available, falling back to S&P 500")
        prices = download_benchmark(start_date, end_date, BENCHMARK_TICKER)
        prices.name = 'MSCI_World'
        return prices

    # Create DataFrame with all returns (allows NaN for missing indices)
    returns_df = pd.DataFrame(index_returns)
    all_dates = returns_df.index.sort_values()

    logger.info(f"Index data ranges:")
    for col in returns_df.columns:
        valid = returns_df[col].dropna()
        logger.info(f"  {col}: {valid.index.min().strftime('%Y-%m')} to {valid.index.max().strftime('%Y-%m')} ({len(valid)} months)")

    # Compute weighted composite returns for each month
    # Use dynamic weighting based on available indices per month
    composite_returns = pd.Series(index=all_dates, dtype=float)

    for date in all_dates:
        row = returns_df.loc[date]
        available = row.dropna()

        if len(available) == 0:
            composite_returns.loc[date] = np.nan
            continue

        # Compute weights for available indices
        total_weight = sum(MSCI_WORLD_PROXIES.get(ticker, 0) for ticker in available.index)
        if total_weight == 0:
            composite_returns.loc[date] = np.nan
            continue

        # Weighted average of available indices
        weighted_return = 0.0
        for ticker in available.index:
            weight = MSCI_WORLD_PROXIES.get(ticker, 0) / total_weight
            weighted_return += available[ticker] * weight

        composite_returns.loc[date] = weighted_return

    # Remove any remaining NaN values
    composite_returns = composite_returns.dropna()
    composite_returns.name = 'MSCI_World'

    if len(composite_returns) < 12:
        logger.warning("Too few valid dates, falling back to S&P 500")
        prices = download_benchmark(start_date, end_date, BENCHMARK_TICKER)
        prices.name = 'MSCI_World'
        return prices

    logger.info(f"MSCI World proxy: {len(composite_returns)} months")
    logger.info(f"Date range: {composite_returns.index.min().strftime('%Y-%m')} to {composite_returns.index.max().strftime('%Y-%m')}")
    logger.info(f"Annualized return: {composite_returns.mean() * 12 * 100:.2f}%")

    # Convert returns to price series (starting at 100)
    composite_prices = (1 + composite_returns).cumprod() * 100
    # Add a starting price point
    first_date = composite_prices.index[0] - pd.DateOffset(months=1)
    composite_prices.loc[first_date] = 100.0
    composite_prices = composite_prices.sort_index()
    composite_prices.name = 'MSCI_World'

    return composite_prices


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
        ticker = yf.Ticker('^IRX')
        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if data.empty:
            raise RuntimeError("No ^IRX data returned")

        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            rf_daily = data['Close'].iloc[:, 0].copy()
        elif 'Close' in data.columns:
            rf_daily = data['Close'].copy()
        else:
            rf_daily = data.iloc[:, 0].copy()

        if isinstance(rf_daily, pd.DataFrame):
            rf_daily = rf_daily.iloc[:, 0]

        # Convert to numeric
        rf_daily = pd.to_numeric(rf_daily, errors='coerce')

        # Resample to month-end
        rf_daily.index = pd.to_datetime(rf_daily.index)
        rf_monthly = rf_daily.resample('ME').last()

        # Convert from annual percentage to monthly decimal
        # ^IRX is quoted as percentage (e.g., 4.5 means 4.5%)
        # Monthly rate = (1 + annual_rate/100)^(1/12) - 1
        rf_monthly_decimal = (1 + rf_monthly / 100) ** (1/12) - 1
        rf_monthly_decimal.name = 'RF_Rate'

        # Remove timezone info if present
        if hasattr(rf_monthly_decimal.index, 'tz') and rf_monthly_decimal.index.tz is not None:
            rf_monthly_decimal.index = rf_monthly_decimal.index.tz_localize(None)

        logger.info(f"Risk-free rate: {len(rf_monthly_decimal)} months")
        logger.info(f"Average monthly RF: {rf_monthly_decimal.mean()*100:.4f}%")

        return rf_monthly_decimal

    except Exception as e:
        logger.warning(f"Error downloading risk-free rate: {e}")
        logger.warning("Using synthetic 2% annual rate as fallback")

        # Create synthetic risk-free rate
        date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
        annual_rate = 2.0  # 2% annual
        monthly_rate = (1 + annual_rate / 100) ** (1/12) - 1
        rf_monthly_decimal = pd.Series(monthly_rate, index=date_range, name='RF_Rate')

        return rf_monthly_decimal


def compute_returns(prices):
    """
    Compute simple monthly returns.

    Args:
        prices: DataFrame of monthly prices

    Returns:
        pd.DataFrame: Monthly returns (first row is NaN, dropped)
    """
    logger.info("Computing monthly returns...")
    returns = prices.pct_change()
    returns = returns.iloc[1:]  # Drop first NaN row
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

    # Handle DataFrame vs Series for rf_rate
    if isinstance(rf_rate, pd.DataFrame):
        rf_rate = rf_rate.iloc[:, 0]

    # Remove timezone info from both
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)
    if hasattr(rf_rate.index, 'tz') and rf_rate.index.tz is not None:
        rf_rate = rf_rate.copy()
        rf_rate.index = rf_rate.index.tz_localize(None)

    # Align dates
    common_dates = returns.index.intersection(rf_rate.index)
    logger.info(f"Common dates for excess returns: {len(common_dates)}")

    # Subtract risk-free rate from each column
    excess_returns = returns.loc[common_dates].subtract(rf_rate.loc[common_dates], axis=0)

    return excess_returns


def compute_rolling_betas(stock_excess_returns, benchmark_excess_returns, window=ROLLING_WINDOW):
    """
    Compute rolling betas for each stock versus the benchmark.

    Beta = Cov(stock_excess, benchmark_excess) / Var(benchmark_excess)

    Args:
        stock_excess_returns: DataFrame of stock excess returns
        benchmark_excess_returns: Series of benchmark excess returns
        window: Rolling window size in months

    Returns:
        pd.DataFrame: Rolling betas for each stock
    """
    logger.info(f"Computing rolling {window}-month betas...")

    # Align dates
    common_dates = stock_excess_returns.index.intersection(benchmark_excess_returns.index)
    stock_rets = stock_excess_returns.loc[common_dates]
    bench_rets = benchmark_excess_returns.loc[common_dates]

    logger.info(f"Computing betas for {len(stock_rets.columns)} stocks over {len(common_dates)} months")

    # Initialize output
    betas = pd.DataFrame(index=stock_rets.index, columns=stock_rets.columns, dtype=float)

    # Pre-compute benchmark variance
    bench_var = bench_rets.rolling(window=window, min_periods=MIN_PERIODS_BETA).var()

    # Compute beta for each stock
    for col in stock_rets.columns:
        stock_series = stock_rets[col]
        cov_series = stock_series.rolling(window=window, min_periods=MIN_PERIODS_BETA).cov(bench_rets)
        betas[col] = cov_series / bench_var

    # Replace inf with NaN
    betas = betas.replace([np.inf, -np.inf], np.nan)

    # Summary statistics
    valid_betas = betas.notna().sum().sum()
    total_cells = betas.shape[0] * betas.shape[1]
    pct_valid = 100 * valid_betas / total_cells if total_cells > 0 else 0

    logger.info(f"Betas shape: {betas.shape}")
    logger.info(f"Valid beta values: {valid_betas:,} of {total_cells:,} ({pct_valid:.1f}%)")

    # Per-month statistics
    valid_per_month = betas.notna().sum(axis=1)
    logger.info(f"Stocks with valid beta per month: min={valid_per_month.min()}, "
                f"max={valid_per_month.max()}, mean={valid_per_month.mean():.1f}")

    return betas


def save_data(data, filename):
    """Save DataFrame to CSV in data directory."""
    ensure_directories()
    filepath = os.path.join(DATA_DIR, filename)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data.to_csv(filepath)
    logger.info(f"Saved {filename}: {data.shape}")


def main():
    """Main data loading pipeline."""
    logger.info("=" * 60)
    logger.info("Starting BAB Data Loader")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("=" * 60)

    ensure_directories()

    # Step 1: Get tickers
    tickers = list(dict.fromkeys(MSCI_WORLD_TICKERS))  # Remove duplicates
    logger.info(f"Using {len(tickers)} tickers from curated MSCI World list")

    # Save ticker list
    pd.DataFrame({"Ticker": tickers}).to_csv(
        os.path.join(DATA_DIR, "ticker_list.csv"), index=False
    )

    # Step 2: Download stock prices
    stock_prices = download_monthly_prices(tickers, START_DATE, END_DATE)

    # Step 3: Download benchmark for beta calculation (S&P 500)
    logger.info("Downloading beta calculation benchmark (S&P 500)...")
    beta_benchmark_prices = download_benchmark(START_DATE, END_DATE, BENCHMARK_TICKER)

    # Step 4: Download MSCI World proxy for performance comparison
    msci_world_prices = download_msci_world_proxy(START_DATE, END_DATE)

    # Step 5: Download risk-free rate
    rf_rate = download_risk_free_rate(START_DATE, END_DATE)

    # Step 6: Compute returns
    stock_returns = compute_returns(stock_prices)

    # Beta benchmark returns (S&P 500)
    beta_bench_df = beta_benchmark_prices.to_frame() if isinstance(beta_benchmark_prices, pd.Series) else beta_benchmark_prices
    beta_benchmark_returns = compute_returns(beta_bench_df).iloc[:, 0]
    beta_benchmark_returns.name = 'Beta_Benchmark'

    # MSCI World proxy returns
    msci_df = msci_world_prices.to_frame() if isinstance(msci_world_prices, pd.Series) else msci_world_prices
    msci_world_returns = compute_returns(msci_df).iloc[:, 0]
    msci_world_returns.name = 'MSCI_World'

    # Step 7: Compute excess returns
    stock_excess_returns = compute_excess_returns(stock_returns, rf_rate)

    # Beta benchmark excess returns (used for beta calculation)
    beta_benchmark_excess = compute_excess_returns(beta_benchmark_returns.to_frame(), rf_rate).iloc[:, 0]
    beta_benchmark_excess.name = 'Beta_Benchmark'

    # MSCI World excess returns (used for performance comparison)
    msci_world_excess = compute_excess_returns(msci_world_returns.to_frame(), rf_rate).iloc[:, 0]
    msci_world_excess.name = 'MSCI_World'

    # Step 8: Compute rolling betas (vs S&P 500)
    logger.info(f"Beta estimation: {ROLLING_WINDOW}-month rolling window, min {MIN_PERIODS_BETA} periods")
    logger.info("Using S&P 500 as beta benchmark (highly correlated with MSCI World)")
    rolling_betas = compute_rolling_betas(stock_excess_returns, beta_benchmark_excess)

    # Step 9: Save all outputs
    logger.info("Saving all data files...")

    # Combine stock and MSCI World prices
    all_prices = stock_prices.copy()
    all_prices['MSCI_World'] = msci_world_prices
    save_data(all_prices, 'monthly_prices.csv')

    # Combine stock and MSCI World returns
    all_returns = stock_returns.copy()
    all_returns['MSCI_World'] = msci_world_returns
    save_data(all_returns, 'monthly_returns.csv')

    # Combine stock and MSCI World excess returns
    all_excess_returns = stock_excess_returns.copy()
    all_excess_returns['MSCI_World'] = msci_world_excess
    save_data(all_excess_returns, 'monthly_excess_returns.csv')

    # Save risk-free rate
    save_data(rf_rate.to_frame(), 'risk_free_rate.csv')

    # Save rolling betas
    save_data(rolling_betas, 'rolling_betas.csv')

    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info(f"Output files saved to: {DATA_DIR}")
    logger.info("=" * 60)

    # Print summary
    print("\n=== Data Summary ===")
    print(f"Date range: {all_prices.index.min().strftime('%Y-%m-%d')} to {all_prices.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of stocks: {len(stock_prices.columns)}")
    print(f"Total months: {len(all_prices)}")

    # Check valid betas in recent months
    recent_betas = rolling_betas.tail(12)
    avg_valid = recent_betas.notna().sum(axis=1).mean()
    print(f"Avg stocks with valid betas (last 12 months): {avg_valid:.0f}")

    if avg_valid < 10:
        logger.warning("Very few stocks have valid betas! Check date range and data.")


if __name__ == '__main__':
    main()
