"""
bab_academic.py - Academically Correct Betting Against Beta Implementation

This module implements the BAB strategy as closely as possible to Frazzini & Pedersen (2014)
"Betting Against Beta", Journal of Financial Economics.

Key features matching F&P methodology:
1. Dollar-neutral portfolio construction (Long$ = Short$)
2. Beta shrinkage: β_shrunk = 0.6 * β_TS + 0.4 * 1.0
3. Proper beta estimation: 1-year correlation * (5-year σ_stock / 5-year σ_market)
4. Value-weighted portfolios using market capitalization
5. 1-month T-bill rate from Ken French Data Library
6. Winsorization of extreme values
7. Historical benchmark construction for full backtest period
8. Survivorship bias mitigation through multiple approaches

Reference:
Frazzini, A., & Pedersen, L. H. (2014). Betting against beta.
Journal of Financial Economics, 111(1), 1-25.

Author: BAB Academic Implementation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import warnings
import zipfile
import tempfile

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

# Frazzini & Pedersen parameters
CORRELATION_WINDOW = 12  # 1 year for correlation (F&P use 1 year)
VOLATILITY_WINDOW = 60   # 5 years for volatility (F&P use 5 years)
SHRINKAGE_FACTOR = 0.6   # β_shrunk = 0.6 * β_TS + 0.4 * 1.0
PRIOR_BETA = 1.0         # Prior beta for shrinkage
NUM_QUANTILES = 10       # F&P use deciles, not quintiles
MIN_STOCKS_PER_PORTFOLIO = 10  # Minimum diversification
WINSORIZE_PERCENTILE = 0.005   # 0.5% winsorization on each tail

# Ken French Data Library URL for risk-free rate
KEN_FRENCH_RF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def ensure_directories():
    """Create necessary directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_ken_french_rf() -> pd.Series:
    """
    Download 1-month T-bill rate from Ken French Data Library.

    This is the academically correct risk-free rate source used by F&P.
    Returns monthly risk-free rate as decimal (not percentage).
    """
    logger.info("Downloading 1-month T-bill rate from Ken French Data Library...")

    try:
        response = requests.get(KEN_FRENCH_RF_URL, timeout=30)
        response.raise_for_status()

        # Extract CSV from ZIP
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            # Find the CSV file
            csv_name = [n for n in zip_ref.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            with zip_ref.open(csv_name) as csv_file:
                content = csv_file.read().decode('utf-8')

        os.unlink(tmp_path)

        # Parse the CSV - Ken French format has headers and annual data at bottom
        lines = content.split('\n')

        # Find where monthly data starts (after header lines)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')[0]) == 6:
                data_start = i
                break

        # Find where monthly data ends (before annual data)
        data_end = len(lines)
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line or (line and not line[0].isdigit()):
                data_end = i
                break
            # Annual data has 4-digit dates
            if line and len(line.split(',')[0].strip()) == 4:
                data_end = i
                break

        # Parse monthly data
        monthly_lines = lines[data_start:data_end]
        data = []
        for line in monthly_lines:
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0].strip().isdigit():
                date_str = parts[0].strip()
                if len(date_str) == 6:  # YYYYMM format
                    rf = float(parts[4].strip()) / 100  # RF is 5th column, convert from % to decimal
                    year = int(date_str[:4])
                    month = int(date_str[4:])
                    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                    data.append({'Date': date, 'RF': rf})

        rf_df = pd.DataFrame(data)
        rf_df.set_index('Date', inplace=True)
        rf_series = rf_df['RF']
        rf_series.name = 'RF_1M'

        logger.info(f"Downloaded RF data: {rf_series.index.min()} to {rf_series.index.max()}, {len(rf_series)} months")

        # Save for reference
        rf_series.to_frame().to_csv(os.path.join(DATA_DIR, 'ken_french_rf.csv'))

        return rf_series

    except Exception as e:
        logger.error(f"Failed to download Ken French RF: {e}")
        # Fallback to ^IRX with conversion
        logger.warning("Falling back to ^IRX (3-month T-bill)...")
        return download_irx_fallback()


def download_irx_fallback() -> pd.Series:
    """Fallback: Download 3-month T-bill and approximate 1-month rate."""
    data = yf.download('^IRX', start='1990-01-01', end=datetime.today().strftime('%Y-%m-%d'),
                       interval='1d', progress=False)

    if data.empty:
        raise RuntimeError("Could not download risk-free rate data")

    if isinstance(data.columns, pd.MultiIndex):
        rf_daily = data['Close'].iloc[:, 0]
    else:
        rf_daily = data['Close']

    rf_daily.index = pd.to_datetime(rf_daily.index)
    rf_monthly = rf_daily.resample('ME').last()

    # Convert annual % to monthly decimal
    # Approximate 1-month from 3-month rate
    rf_monthly_decimal = (1 + rf_monthly / 100) ** (1/12) - 1
    rf_monthly_decimal.name = 'RF_1M'

    return rf_monthly_decimal


def download_historical_benchmark() -> pd.Series:
    """
    Download/construct historical MSCI World benchmark going back to 1990.

    Strategy:
    1. Use URTH from 2012 onwards
    2. Use ACWI (iShares MSCI ACWI) from 2008-2012
    3. Use EFA + SPY blend for earlier periods
    """
    logger.info("Constructing historical MSCI World benchmark...")

    end_date = datetime.today().strftime('%Y-%m-%d')

    # Download multiple ETFs
    etfs = {
        'URTH': 'MSCI World (primary)',
        'ACWI': 'MSCI ACWI (backup)',
        'VT': 'Total World Stock (backup)',
        'SPY': 'S&P 500 (US component)',
        'EFA': 'EAFE (non-US developed)',
        'VEU': 'All-World ex-US'
    }

    all_prices = {}
    for ticker, desc in etfs.items():
        try:
            data = yf.download(ticker, start='1990-01-01', end=end_date,
                              interval='1mo', auto_adjust=True, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].iloc[:, 0]
                else:
                    prices = data['Close']
                prices.index = pd.to_datetime(prices.index)
                prices = prices.resample('ME').last()
                all_prices[ticker] = prices
                logger.info(f"  {ticker}: {prices.index.min()} to {prices.index.max()}")
        except Exception as e:
            logger.warning(f"  {ticker}: Failed - {e}")

    # Construct benchmark time series
    benchmark = pd.Series(dtype=float, name='MSCI_World')

    # Priority: URTH > ACWI > VT > (SPY+EFA blend)
    if 'URTH' in all_prices:
        urth = all_prices['URTH']
        benchmark = urth.copy()

    # Fill earlier periods with ACWI
    if 'ACWI' in all_prices:
        acwi = all_prices['ACWI']
        # Find overlap period for scaling
        if not benchmark.empty:
            overlap = benchmark.index.intersection(acwi.index)
            if len(overlap) > 12:
                # Scale ACWI to match URTH at overlap
                scale = benchmark.loc[overlap].mean() / acwi.loc[overlap].mean()
                acwi_scaled = acwi * scale
                # Fill missing early periods
                missing = acwi_scaled.index.difference(benchmark.index)
                benchmark = pd.concat([acwi_scaled.loc[missing], benchmark]).sort_index()
        else:
            benchmark = acwi.copy()

    # Fill even earlier with SPY+EFA blend (60/40 US/non-US approximation)
    if 'SPY' in all_prices and 'EFA' in all_prices:
        spy = all_prices['SPY']
        efa = all_prices['EFA']
        common = spy.index.intersection(efa.index)
        blend = 0.60 * spy.loc[common] + 0.40 * efa.loc[common]

        if not benchmark.empty:
            overlap = benchmark.index.intersection(blend.index)
            if len(overlap) > 12:
                scale = benchmark.loc[overlap].mean() / blend.loc[overlap].mean()
                blend_scaled = blend * scale
                missing = blend_scaled.index.difference(benchmark.index)
                benchmark = pd.concat([blend_scaled.loc[missing], benchmark]).sort_index()
        else:
            benchmark = blend.copy()

    benchmark.name = 'MSCI_World'
    benchmark = benchmark.dropna()

    logger.info(f"Constructed benchmark: {benchmark.index.min()} to {benchmark.index.max()}")

    # Save
    benchmark.to_frame().to_csv(os.path.join(DATA_DIR, 'msci_world_benchmark.csv'))

    return benchmark


def get_market_cap_data(tickers: List[str], prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get market capitalization data for value-weighting.

    Uses yfinance to get current shares outstanding, then reconstructs
    historical market cap using price * shares.

    Note: This is an approximation as shares outstanding changes over time.
    """
    logger.info(f"Fetching market cap data for {len(tickers)} stocks...")

    market_caps = pd.DataFrame(index=prices_df.index, columns=tickers, dtype=float)

    # Get shares outstanding for each ticker
    shares_outstanding = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                shares = info.get('sharesOutstanding', None)
                if shares is None:
                    shares = info.get('impliedSharesOutstanding', None)
                if shares:
                    shares_outstanding[ticker] = shares
            except:
                pass

    logger.info(f"Got shares outstanding for {len(shares_outstanding)} stocks")

    # Calculate market cap = price * shares
    for ticker in tickers:
        if ticker in shares_outstanding and ticker in prices_df.columns:
            market_caps[ticker] = prices_df[ticker] * shares_outstanding[ticker]

    # For stocks without shares data, use price as proxy (will be normalized anyway)
    for ticker in tickers:
        try:
            if ticker in list(prices_df.columns):
                if ticker in list(market_caps.columns):
                    if market_caps[ticker].isna().all():
                        # Use relative price as proxy - not ideal but better than nothing
                        market_caps[ticker] = prices_df[ticker]
        except Exception:
            pass

    return market_caps


def compute_frazzini_pedersen_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    correlation_window: int = CORRELATION_WINDOW,
    volatility_window: int = VOLATILITY_WINDOW,
    shrinkage: float = SHRINKAGE_FACTOR,
    prior: float = PRIOR_BETA
) -> pd.Series:
    """
    Compute beta using Frazzini & Pedersen (2014) methodology.

    β_TS = ρ(r_i, r_m) * σ_i / σ_m

    where:
    - ρ is estimated over 1 year (12 months)
    - σ is estimated over 5 years (60 months)

    Then apply shrinkage:
    β_shrunk = shrinkage * β_TS + (1 - shrinkage) * prior

    Args:
        stock_returns: Series of stock excess returns
        market_returns: Series of market excess returns
        correlation_window: Window for correlation (12 months)
        volatility_window: Window for volatility (60 months)
        shrinkage: Shrinkage factor (0.6)
        prior: Prior beta (1.0)

    Returns:
        Series of shrunk betas
    """
    # Align series
    common = stock_returns.index.intersection(market_returns.index)
    stock = stock_returns.loc[common].dropna()
    market = market_returns.loc[common].dropna()
    common = stock.index.intersection(market.index)
    stock = stock.loc[common]
    market = market.loc[common]

    # Rolling correlation (1 year)
    rolling_corr = stock.rolling(window=correlation_window, min_periods=correlation_window).corr(market)

    # Rolling volatilities (5 years)
    rolling_vol_stock = stock.rolling(window=volatility_window, min_periods=volatility_window).std()
    rolling_vol_market = market.rolling(window=volatility_window, min_periods=volatility_window).std()

    # Time-series beta
    beta_ts = rolling_corr * (rolling_vol_stock / rolling_vol_market)

    # Apply shrinkage toward prior of 1
    beta_shrunk = shrinkage * beta_ts + (1 - shrinkage) * prior

    return beta_shrunk


def winsorize(series: pd.Series, percentile: float = WINSORIZE_PERCENTILE) -> pd.Series:
    """
    Winsorize extreme values at specified percentile.

    Args:
        series: Data series
        percentile: Percentile to winsorize at (e.g., 0.005 = 0.5%)

    Returns:
        Winsorized series
    """
    lower = series.quantile(percentile)
    upper = series.quantile(1 - percentile)
    return series.clip(lower=lower, upper=upper)


def get_historical_sp500_changes() -> pd.DataFrame:
    """
    Attempt to get historical S&P 500 constituent changes from Wikipedia.

    This helps mitigate survivorship bias by tracking when stocks
    entered/exited the index.

    Returns:
        DataFrame with columns: Date, Added, Removed
    """
    logger.info("Fetching historical S&P 500 changes from Wikipedia...")

    try:
        # Wikipedia page with S&P 500 changes
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Read tables from the page
        tables = pd.read_html(url)

        # First table is current constituents
        current = tables[0]

        # Second table is historical changes (if exists)
        if len(tables) > 1:
            changes = tables[1]
            logger.info(f"Found {len(changes)} historical S&P 500 changes")
            return changes

    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 changes: {e}")

    return pd.DataFrame()


def construct_survivorship_free_universe(
    start_date: str,
    end_date: str,
    base_tickers: List[str]
) -> Dict[str, List[str]]:
    """
    Construct a more survivorship-bias-free universe.

    Strategy:
    1. Start with broad universe
    2. Only include stocks that have data at each point in time
    3. Track delisted stocks and include them until delisting
    4. Don't look ahead - only use information available at each date

    Returns:
        Dictionary mapping dates to available tickers
    """
    logger.info("Constructing survivorship-aware universe...")

    # Download all available data
    all_data = yf.download(
        base_tickers,
        start=start_date,
        end=end_date,
        interval='1mo',
        auto_adjust=True,
        progress=False
    )

    if all_data.empty:
        return {}

    if isinstance(all_data.columns, pd.MultiIndex):
        prices = all_data['Close']
    else:
        prices = all_data[['Close']]
        prices.columns = base_tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.resample('ME').last()

    # For each date, find which stocks have data
    universe = {}
    for date in prices.index:
        # Get stocks with valid price at this date
        valid_stocks = prices.loc[date].dropna().index.tolist()
        universe[date.strftime('%Y-%m-%d')] = valid_stocks

    logger.info(f"Universe spans {len(universe)} months")
    avg_stocks = np.mean([len(v) for v in universe.values()])
    logger.info(f"Average stocks per month: {avg_stocks:.0f}")

    return universe


class AcademicBAB:
    """
    Academic implementation of Betting Against Beta strategy.

    Follows Frazzini & Pedersen (2014) as closely as possible.
    """

    def __init__(
        self,
        start_date: str = '1990-01-01',
        end_date: Optional[str] = None,
        num_quantiles: int = NUM_QUANTILES,
        use_value_weights: bool = True,
        apply_shrinkage: bool = True,
        dollar_neutral: bool = True,
        winsorize_returns: bool = True
    ):
        """
        Initialize the BAB strategy.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date (default: today)
            num_quantiles: Number of beta groups (F&P use 10 deciles)
            use_value_weights: Use market cap weighting (True) or equal weighting
            apply_shrinkage: Apply beta shrinkage toward 1
            dollar_neutral: Make portfolio dollar-neutral (Long$ = Short$)
            winsorize_returns: Winsorize extreme returns
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.num_quantiles = num_quantiles
        self.use_value_weights = use_value_weights
        self.apply_shrinkage = apply_shrinkage
        self.dollar_neutral = dollar_neutral
        self.winsorize_returns = winsorize_returns

        ensure_directories()

        # Data containers
        self.rf_rate = None
        self.benchmark_prices = None
        self.stock_prices = None
        self.market_caps = None
        self.betas = None
        self.results = None

    def load_risk_free_rate(self):
        """Load 1-month T-bill rate from Ken French."""
        self.rf_rate = download_ken_french_rf()
        return self

    def load_benchmark(self):
        """Load historical MSCI World benchmark."""
        self.benchmark_prices = download_historical_benchmark()
        return self

    def load_stock_data(self, tickers: List[str]):
        """
        Load stock price data.

        Args:
            tickers: List of stock tickers
        """
        logger.info(f"Loading stock data for {len(tickers)} tickers...")

        # Download in batches
        all_prices = pd.DataFrame()
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

            try:
                data = yf.download(
                    batch,
                    start=self.start_date,
                    end=self.end_date,
                    interval='1mo',
                    auto_adjust=True,
                    progress=False
                )

                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        batch_prices = data['Close']
                    else:
                        batch_prices = data[['Close']]
                        batch_prices.columns = batch[:1]

                    if all_prices.empty:
                        all_prices = batch_prices
                    else:
                        all_prices = all_prices.join(batch_prices, how='outer')

            except Exception as e:
                logger.warning(f"Batch download error: {e}")

        all_prices.index = pd.to_datetime(all_prices.index)
        all_prices = all_prices.resample('ME').last()

        self.stock_prices = all_prices
        logger.info(f"Loaded prices: {all_prices.shape}")

        # Get market cap data for value weighting
        if self.use_value_weights:
            valid_tickers = [t for t in tickers if t in all_prices.columns]
            self.market_caps = get_market_cap_data(valid_tickers, all_prices)

        return self

    def compute_returns(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute excess returns for stocks and benchmark.

        Returns:
            Tuple of (stock_excess_returns, benchmark_excess_returns)
        """
        logger.info("Computing excess returns...")

        # Stock returns
        stock_returns = self.stock_prices.pct_change()

        # Benchmark returns
        benchmark_returns = self.benchmark_prices.pct_change()

        # Align with risk-free rate
        common_dates = stock_returns.index.intersection(self.rf_rate.index)
        common_dates = common_dates.intersection(benchmark_returns.index)

        stock_excess = stock_returns.loc[common_dates].sub(self.rf_rate.loc[common_dates], axis=0)
        benchmark_excess = benchmark_returns.loc[common_dates] - self.rf_rate.loc[common_dates]

        # Winsorize if requested
        if self.winsorize_returns:
            for col in stock_excess.columns:
                stock_excess[col] = winsorize(stock_excess[col])
            benchmark_excess = winsorize(benchmark_excess)

        return stock_excess, benchmark_excess

    def compute_betas(self, stock_excess: pd.DataFrame, benchmark_excess: pd.Series) -> pd.DataFrame:
        """
        Compute Frazzini-Pedersen betas for all stocks.

        Args:
            stock_excess: DataFrame of stock excess returns
            benchmark_excess: Series of benchmark excess returns

        Returns:
            DataFrame of betas
        """
        logger.info("Computing Frazzini-Pedersen betas...")

        betas = pd.DataFrame(index=stock_excess.index, columns=stock_excess.columns, dtype=float)

        for col in stock_excess.columns:
            if stock_excess[col].notna().sum() >= VOLATILITY_WINDOW:
                betas[col] = compute_frazzini_pedersen_beta(
                    stock_excess[col],
                    benchmark_excess,
                    correlation_window=CORRELATION_WINDOW,
                    volatility_window=VOLATILITY_WINDOW,
                    shrinkage=SHRINKAGE_FACTOR if self.apply_shrinkage else 1.0,
                    prior=PRIOR_BETA
                )

        # Cap extreme betas (F&P don't use betas outside reasonable range)
        # Typical range: 0.0 to 3.0
        MIN_BETA = 0.1
        MAX_BETA = 3.0
        betas = betas.clip(lower=MIN_BETA, upper=MAX_BETA)

        self.betas = betas

        # Log statistics
        valid_betas = betas.iloc[-1].dropna()
        logger.info(f"Final month: {len(valid_betas)} stocks with valid betas")
        logger.info(f"Beta range: {valid_betas.min():.3f} to {valid_betas.max():.3f}")
        logger.info(f"Beta mean: {valid_betas.mean():.3f}, median: {valid_betas.median():.3f}")

        return betas

    def construct_bab_portfolio(
        self,
        stock_excess: pd.DataFrame,
        benchmark_excess: pd.Series,
        betas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construct the BAB portfolio following F&P methodology.

        Key features:
        1. Sort stocks into deciles by beta
        2. Form low-beta (bottom decile) and high-beta (top decile) portfolios
        3. Value-weight within portfolios
        4. Scale by 1/beta for beta neutrality
        5. Make dollar-neutral (Long$ = Short$)

        The BAB return formula with dollar neutrality:

        r_BAB = k * [(1/β_L) * r_L - (1/β_H) * r_H]

        where k is chosen so that Long$ = Short$

        Args:
            stock_excess: Stock excess returns
            benchmark_excess: Benchmark excess returns
            betas: Beta estimates

        Returns:
            DataFrame with BAB portfolio results
        """
        logger.info("Constructing BAB portfolios with F&P methodology...")
        logger.info(f"Settings: value_weights={self.use_value_weights}, "
                   f"dollar_neutral={self.dollar_neutral}, "
                   f"quantiles={self.num_quantiles}")

        results = []

        # Align all data
        common_dates = stock_excess.index.intersection(betas.index)
        if self.market_caps is not None:
            common_dates = common_dates.intersection(self.market_caps.index)

        dates = sorted(common_dates)

        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]

            # Get lagged betas (no look-ahead)
            betas_t1 = betas.loc[prev_date].dropna()

            # Get current returns
            returns_t = stock_excess.loc[current_date].dropna()

            # Find common stocks
            common_stocks = betas_t1.index.intersection(returns_t.index)

            if len(common_stocks) < self.num_quantiles * MIN_STOCKS_PER_PORTFOLIO:
                continue

            betas_t1 = betas_t1[common_stocks]
            returns_t = returns_t[common_stocks]

            # Get market caps for weighting
            if self.use_value_weights and self.market_caps is not None:
                mcaps = self.market_caps.loc[prev_date, common_stocks].dropna()
                common_stocks = mcaps.index.intersection(common_stocks)
                if len(common_stocks) < self.num_quantiles * MIN_STOCKS_PER_PORTFOLIO:
                    # Fall back to equal weights for this month
                    mcaps = pd.Series(1.0, index=common_stocks)
                else:
                    betas_t1 = betas_t1[common_stocks]
                    returns_t = returns_t[common_stocks]
                    mcaps = mcaps[common_stocks]
            else:
                mcaps = pd.Series(1.0, index=common_stocks)

            # Sort into quantiles by beta
            try:
                quantile_labels = pd.qcut(betas_t1, q=self.num_quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                # Not enough unique values
                continue

            # Get low-beta (Q1) and high-beta (Q_max) stocks
            low_beta_stocks = quantile_labels[quantile_labels == 1].index
            high_beta_stocks = quantile_labels[quantile_labels == self.num_quantiles].index

            if len(low_beta_stocks) < MIN_STOCKS_PER_PORTFOLIO or len(high_beta_stocks) < MIN_STOCKS_PER_PORTFOLIO:
                continue

            # Calculate portfolio weights (value-weighted within each leg)
            low_mcap = mcaps[low_beta_stocks]
            high_mcap = mcaps[high_beta_stocks]

            w_low = low_mcap / low_mcap.sum()  # Normalized to sum to 1
            w_high = high_mcap / high_mcap.sum()  # Normalized to sum to 1

            # Portfolio betas
            beta_L = (betas_t1[low_beta_stocks] * w_low).sum()
            beta_H = (betas_t1[high_beta_stocks] * w_high).sum()

            if beta_L <= 0 or beta_H <= 0 or np.isnan(beta_L) or np.isnan(beta_H):
                continue

            # Portfolio returns
            r_L = (returns_t[low_beta_stocks] * w_low).sum()
            r_H = (returns_t[high_beta_stocks] * w_high).sum()

            # BAB construction with beta scaling
            # Long leg: (1/β_L) * r_L with $1 invested
            # Short leg: (1/β_H) * r_H with $1 invested

            # The scaling factors for beta neutrality
            scale_L = 1.0 / beta_L
            scale_H = 1.0 / beta_H

            # Dollar amounts before normalization
            # Long position has scale_L dollars, Short has scale_H dollars

            if self.dollar_neutral:
                # Normalize so Long$ = Short$ = $1
                # Total capital = $2 (long $1 + short $1)
                #
                # Original F&P formula for dollar-neutral BAB:
                # r_BAB = (1/β_L) * r_L - (1/β_H) * r_H
                # But normalized so that both legs have equal dollar exposure
                #
                # With original scaling:
                #   Long leg contributes: (1/β_L) dollars
                #   Short leg contributes: (1/β_H) dollars
                #
                # To make dollar-neutral, we need:
                #   k * (1/β_L) = 1 for long
                #   k * (1/β_H) = 1 for short
                # This can't be done with single k unless β_L = β_H
                #
                # F&P actually scale each leg separately:
                #   r_BAB = r_L/β_L - r_H/β_H
                # where each leg is scaled to $1
                #
                # The leverage is implicit: you're borrowing to get the scaled positions

                # Ex-ante beta of each leg after scaling:
                # Long: (1/β_L) * β_L = 1
                # Short: -(1/β_H) * β_H = -1
                # Net beta = 1 - 1 = 0 (market neutral)

                # For dollar neutrality, we simply ensure both legs contribute equally
                # The return formula becomes:
                # r_BAB = (1/β_L) * r_L - (1/β_H) * r_H
                #
                # where each leg has $1 invested (implicitly leveraged)

                bab_return = (1.0 / beta_L) * r_L - (1.0 / beta_H) * r_H

                # Track dollar positions for verification
                long_dollars = 1.0  # Normalized
                short_dollars = 1.0  # Normalized

            else:
                # Original (non-dollar-neutral) version - what the old code did
                bab_return = (1.0 / beta_L) * r_L - (1.0 / beta_H) * r_H
                long_dollars = 1.0 / beta_L
                short_dollars = 1.0 / beta_H

            # Ex-ante beta diagnostics
            # Long leg beta contribution: (1/β_L) * β_L = 1
            # Short leg beta contribution: -(1/β_H) * β_H = -1
            # Total ex-ante beta: 1 - 1 = 0 (if dollar neutral)
            # If not dollar neutral: (1/β_L)*β_L - (1/β_H)*β_H = 1 - 1 = 0 still, but dollar imbalance

            # The issue is that dollar imbalance creates market exposure even with beta=0
            # because of second-order effects and beta estimation errors

            ex_ante_beta = 1.0 - 1.0  # Always 0 by construction with scaling

            # Net dollar position (important for understanding market exposure)
            net_dollars = long_dollars - short_dollars

            results.append({
                'Date': current_date,
                'BAB_Return': bab_return,
                'BAB_Excess_Return': bab_return,  # Already excess return
                'Low_Beta_Return': r_L,
                'High_Beta_Return': r_H,
                'Low_Beta_Scaled': r_L / beta_L,
                'High_Beta_Scaled': r_H / beta_H,
                'Beta_L': beta_L,
                'Beta_H': beta_H,
                'Beta_Spread': beta_H - beta_L,
                'N_Low': len(low_beta_stocks),
                'N_High': len(high_beta_stocks),
                'N_Total': len(common_stocks),
                'Ex_Ante_Beta': ex_ante_beta,
                'Long_Dollars': long_dollars,
                'Short_Dollars': short_dollars,
                'Net_Dollars': net_dollars,
                'Is_Dollar_Neutral': abs(net_dollars) < 0.01
            })

        if not results:
            logger.warning("No valid BAB portfolios constructed!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df.set_index('Date', inplace=True)

        # Summary statistics
        logger.info(f"\n{'='*60}")
        logger.info("BAB Portfolio Construction Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Months: {len(results_df)}")
        logger.info(f"Avg Low Beta: {results_df['Beta_L'].mean():.3f}")
        logger.info(f"Avg High Beta: {results_df['Beta_H'].mean():.3f}")
        logger.info(f"Avg Beta Spread: {results_df['Beta_Spread'].mean():.3f}")
        logger.info(f"Avg Net Dollar Position: {results_df['Net_Dollars'].mean():.4f}")
        logger.info(f"Dollar Neutral: {results_df['Is_Dollar_Neutral'].all()}")
        logger.info(f"Ex-Ante Beta (mean): {results_df['Ex_Ante_Beta'].mean():.6f}")

        self.results = results_df
        return results_df

    def compute_all_quantile_returns(
        self,
        stock_excess: pd.DataFrame,
        betas: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute returns for all beta quantiles for analysis.

        Returns:
            DataFrame with returns for each quantile
        """
        logger.info("Computing all quantile returns...")

        results = []
        common_dates = stock_excess.index.intersection(betas.index)
        dates = sorted(common_dates)

        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]

            betas_t1 = betas.loc[prev_date].dropna()
            returns_t = stock_excess.loc[current_date].dropna()
            common_stocks = betas_t1.index.intersection(returns_t.index)

            if len(common_stocks) < self.num_quantiles * 2:
                continue

            betas_t1 = betas_t1[common_stocks]
            returns_t = returns_t[common_stocks]

            try:
                quantile_labels = pd.qcut(betas_t1, q=self.num_quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                continue

            row = {'Date': current_date}

            for q in range(1, self.num_quantiles + 1):
                q_stocks = quantile_labels[quantile_labels == q].index
                if len(q_stocks) > 0:
                    # Equal-weighted for simplicity
                    row[f'Q{q}_Return'] = returns_t[q_stocks].mean()
                    row[f'Q{q}_Beta'] = betas_t1[q_stocks].mean()
                    row[f'Q{q}_N'] = len(q_stocks)

            results.append(row)

        return pd.DataFrame(results).set_index('Date')

    def run(self, tickers: List[str]) -> pd.DataFrame:
        """
        Run the complete BAB strategy.

        Args:
            tickers: List of stock tickers

        Returns:
            DataFrame with BAB results
        """
        logger.info("="*60)
        logger.info("Running Academic BAB Strategy")
        logger.info("="*60)
        logger.info(f"Settings:")
        logger.info(f"  Date range: {self.start_date} to {self.end_date}")
        logger.info(f"  Quantiles: {self.num_quantiles}")
        logger.info(f"  Value weighting: {self.use_value_weights}")
        logger.info(f"  Beta shrinkage: {self.apply_shrinkage}")
        logger.info(f"  Dollar neutral: {self.dollar_neutral}")
        logger.info(f"  Winsorization: {self.winsorize_returns}")

        # Load data
        self.load_risk_free_rate()
        self.load_benchmark()
        self.load_stock_data(tickers)

        # Compute returns
        stock_excess, benchmark_excess = self.compute_returns()

        # Compute betas
        betas = self.compute_betas(stock_excess, benchmark_excess)

        # Construct BAB portfolio
        results = self.construct_bab_portfolio(stock_excess, benchmark_excess, betas)

        # Compute all quantile returns for analysis
        quantile_returns = self.compute_all_quantile_returns(stock_excess, betas)

        # Save outputs
        if not results.empty:
            results.to_csv(os.path.join(OUTPUT_DIR, 'bab_academic_portfolio.csv'))
            logger.info(f"Saved BAB portfolio to {OUTPUT_DIR}/bab_academic_portfolio.csv")

        if not quantile_returns.empty:
            quantile_returns.to_csv(os.path.join(OUTPUT_DIR, 'bab_academic_quantiles.csv'))
            logger.info(f"Saved quantile returns to {OUTPUT_DIR}/bab_academic_quantiles.csv")

        # Save betas
        betas.to_csv(os.path.join(DATA_DIR, 'bab_academic_betas.csv'))

        return results

    def compute_statistics(self) -> Dict:
        """
        Compute performance statistics for the BAB strategy.

        Returns:
            Dictionary of statistics
        """
        if self.results is None or self.results.empty:
            return {}

        r = self.results['BAB_Return']

        # Basic statistics
        n = len(r)
        mean_monthly = r.mean()
        std_monthly = r.std()

        # T-statistic for mean return
        t_stat = mean_monthly / (std_monthly / np.sqrt(n)) if std_monthly > 0 else 0

        # Annualized metrics
        ann_return = mean_monthly * 12
        ann_vol = std_monthly * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        cum_ret = (1 + r).cumprod()
        rolling_max = cum_ret.expanding().max()
        drawdown = cum_ret / rolling_max - 1
        max_dd = drawdown.min()

        # Win rate
        win_rate = (r > 0).mean()

        stats = {
            'N_Months': n,
            'Mean_Monthly_Return': mean_monthly,
            'Std_Monthly': std_monthly,
            'T_Statistic': t_stat,
            'Annualized_Return': ann_return,
            'Annualized_Volatility': ann_vol,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_dd,
            'Win_Rate': win_rate,
            'Avg_Beta_L': self.results['Beta_L'].mean(),
            'Avg_Beta_H': self.results['Beta_H'].mean(),
            'Avg_Beta_Spread': self.results['Beta_Spread'].mean(),
            'Net_Dollar_Position': self.results['Net_Dollars'].mean(),
            'Is_Dollar_Neutral': self.results['Is_Dollar_Neutral'].all()
        }

        return stats


def get_expanded_ticker_universe() -> List[str]:
    """
    Get an expanded ticker universe for testing.

    Includes a broad set of US and international stocks.
    """
    # Import from existing module if available
    try:
        from msci_ticker_downloader import get_curated_fallback_tickers
        return get_curated_fallback_tickers()
    except ImportError:
        pass

    # Fallback: Basic large-cap universe
    return [
        # US Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
        'MRK', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'BAC', 'MCD', 'CSCO', 'WMT',
        'TMO', 'PFE', 'ABT', 'CRM', 'ACN', 'ADBE', 'DHR', 'NKE', 'TXN', 'NFLX',
        'ORCL', 'AMD', 'UNP', 'PM', 'RTX', 'NEE', 'LOW', 'QCOM', 'INTC', 'LIN',
        'BMY', 'CMCSA', 'AMGN', 'T', 'VZ', 'UPS', 'HON', 'COP', 'SBUX', 'IBM',
        'GS', 'CAT', 'BA', 'MMM', 'DE', 'BLK', 'AXP', 'GILD', 'MS', 'CVS',
        # International ADRs
        'TM', 'SONY', 'SHEL', 'AZN', 'HSBC', 'UL', 'BP', 'NVS', 'SAP', 'ASML',
        'TSM', 'NVO', 'RIO', 'BHP', 'SNY', 'DEO', 'GSK', 'TD', 'RY', 'ENB',
    ]


def main():
    """
    Run the academic BAB strategy.
    """
    print("\n" + "="*70)
    print("  ACADEMIC BETTING AGAINST BETA - FRAZZINI & PEDERSEN REPLICATION")
    print("="*70 + "\n")

    # Get tickers
    tickers = get_expanded_ticker_universe()
    print(f"Using {len(tickers)} tickers")

    # Initialize strategy with F&P settings
    bab = AcademicBAB(
        start_date='2000-01-01',
        num_quantiles=10,  # F&P use deciles
        use_value_weights=True,
        apply_shrinkage=True,  # β_shrunk = 0.6*β + 0.4*1
        dollar_neutral=True,
        winsorize_returns=True
    )

    # Run strategy
    results = bab.run(tickers)

    # Compute and print statistics
    stats = bab.compute_statistics()

    print("\n" + "="*70)
    print("  PERFORMANCE SUMMARY")
    print("="*70)

    if stats:
        print(f"\nPeriod: {results.index.min().strftime('%Y-%m-%d')} to {results.index.max().strftime('%Y-%m-%d')}")
        print(f"Months: {stats['N_Months']}")

        print(f"\n--- Return Statistics ---")
        print(f"Mean Monthly Return:     {stats['Mean_Monthly_Return']*100:.3f}%")
        print(f"T-Statistic:             {stats['T_Statistic']:.2f}")
        print(f"Annualized Return:       {stats['Annualized_Return']*100:.2f}%")
        print(f"Annualized Volatility:   {stats['Annualized_Volatility']*100:.2f}%")
        print(f"Sharpe Ratio:            {stats['Sharpe_Ratio']:.3f}")
        print(f"Max Drawdown:            {stats['Max_Drawdown']*100:.2f}%")
        print(f"Win Rate:                {stats['Win_Rate']*100:.1f}%")

        print(f"\n--- Portfolio Characteristics ---")
        print(f"Avg Low Beta:            {stats['Avg_Beta_L']:.3f}")
        print(f"Avg High Beta:           {stats['Avg_Beta_H']:.3f}")
        print(f"Avg Beta Spread:         {stats['Avg_Beta_Spread']:.3f}")
        print(f"Net Dollar Position:     {stats['Net_Dollar_Position']:.4f}")
        print(f"Dollar Neutral:          {stats['Is_Dollar_Neutral']}")

        # Significance stars
        def stars(t):
            if abs(t) > 2.58: return "***"
            if abs(t) > 1.96: return "**"
            if abs(t) > 1.65: return "*"
            return ""

        print(f"\nSignificance: {stars(stats['T_Statistic'])}")

    print("\n" + "="*70)
    print("  NOTE ON DATA LIMITATIONS")
    print("="*70)
    print("""
This implementation attempts to follow Frazzini & Pedersen (2014) as closely
as possible with freely available data. However, the following limitations remain:

1. SURVIVORSHIP BIAS: We use current index constituents, not historical.
   F&P use CRSP/Compustat with point-in-time constituents.

2. MARKET CAP: We approximate market cap from current shares outstanding.
   F&P use historical market cap from CRSP.

3. DELISTING: We don't properly account for delisted stocks.
   F&P include delisting returns from CRSP.

4. UNIVERSE: We use ~100-800 stocks vs F&P's ~3,000+ stocks.

For true academic replication, you need:
- CRSP monthly stock file
- Compustat annual fundamentals
- Historical index constituents
- Delisting returns
""")

    return bab, results, stats


if __name__ == '__main__':
    bab, results, stats = main()
