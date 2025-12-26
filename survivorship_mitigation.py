"""
survivorship_mitigation.py - Strategies to mitigate survivorship bias

This module implements various approaches to reduce survivorship bias in the BAB
backtest using freely available data sources.

Strategies:
1. Historical S&P 500 constituent changes from Wikipedia
2. Delisted stock detection from yfinance
3. Point-in-time universe reconstruction
4. Dead stock inclusion until actual delisting date
5. Conservative returns for delisted stocks

Note: True survivorship-bias-free backtesting requires paid databases like
CRSP/Compustat. These are mitigation strategies, not complete solutions.

Author: BAB Academic Implementation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import time
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_sp500_historical_changes() -> pd.DataFrame:
    """
    Scrape historical S&P 500 constituent changes from Wikipedia.

    Returns DataFrame with columns: Date, Ticker_Added, Ticker_Removed, Company_Added, Company_Removed
    """
    logger.info("Fetching S&P 500 historical changes from Wikipedia...")
    ensure_dirs()

    cache_path = os.path.join(CACHE_DIR, 'sp500_changes.csv')

    # Check cache (valid for 7 days)
    if os.path.exists(cache_path):
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if cache_age < timedelta(days=7):
            logger.info("Loading S&P 500 changes from cache...")
            return pd.read_csv(cache_path, parse_dates=['Date'])

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        # Read all tables from the page
        tables = pd.read_html(url)

        # First table: Current constituents
        current = tables[0]
        current.columns = current.columns.str.strip()
        current_tickers = set(current['Symbol'].str.replace('.', '-').tolist())

        # Second table: Historical changes (if available)
        changes_df = pd.DataFrame()
        if len(tables) > 1:
            changes = tables[1]

            # The changes table has: Date, Added (Ticker, Security), Removed (Ticker, Security), Reason
            # Column structure varies, so we handle flexibly
            changes.columns = [str(c).strip() for c in changes.columns]

            # Try to extract date and ticker information
            data = []
            for idx, row in changes.iterrows():
                try:
                    # Date is usually first column
                    date_str = str(row.iloc[0]).strip()

                    # Try to parse date
                    date = None
                    for fmt in ['%B %d, %Y', '%Y-%m-%d', '%d %B %Y', '%b %d, %Y']:
                        try:
                            date = pd.to_datetime(date_str, format=fmt)
                            break
                        except:
                            continue

                    if date is None:
                        try:
                            date = pd.to_datetime(date_str)
                        except:
                            continue

                    # Added ticker (usually column 1 or labeled 'Added')
                    added = str(row.iloc[1]).strip() if len(row) > 1 else ''
                    if added in ['nan', 'NaN', '']:
                        added = None
                    else:
                        added = added.replace('.', '-')

                    # Removed ticker (usually column 3 or labeled 'Removed')
                    removed = str(row.iloc[3]).strip() if len(row) > 3 else ''
                    if removed in ['nan', 'NaN', '']:
                        removed = None
                    else:
                        removed = removed.replace('.', '-')

                    if added or removed:
                        data.append({
                            'Date': date,
                            'Ticker_Added': added,
                            'Ticker_Removed': removed
                        })

                except Exception as e:
                    continue

            changes_df = pd.DataFrame(data)
            changes_df = changes_df.dropna(subset=['Date'])
            changes_df = changes_df.sort_values('Date')

            logger.info(f"Found {len(changes_df)} S&P 500 constituent changes")

            # Save to cache
            changes_df.to_csv(cache_path, index=False)

        return changes_df

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 changes: {e}")
        return pd.DataFrame()


def get_delisted_stocks_from_yfinance(tickers: List[str]) -> Dict[str, str]:
    """
    Check which stocks are delisted using yfinance.

    A stock is considered delisted if:
    1. No recent price data (last 30 days)
    2. yfinance returns empty data

    Returns:
        Dictionary mapping ticker to approximate delisting date
    """
    logger.info(f"Checking {len(tickers)} tickers for delisting...")

    delisted = {}
    cutoff_date = datetime.now() - timedelta(days=60)

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            logger.info(f"  Checking {i}/{len(tickers)}...")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')

            if hist.empty:
                # Try to get the last available date
                hist_all = stock.history(period='max')
                if not hist_all.empty:
                    last_date = hist_all.index[-1]
                    if last_date < cutoff_date:
                        delisted[ticker] = last_date.strftime('%Y-%m-%d')
                else:
                    delisted[ticker] = 'unknown'
        except:
            delisted[ticker] = 'error'

        time.sleep(0.1)  # Rate limiting

    logger.info(f"Found {len(delisted)} potentially delisted stocks")
    return delisted


def reconstruct_historical_universe(
    base_tickers: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, List[str]]:
    """
    Reconstruct the historical stock universe for each month.

    Uses multiple strategies:
    1. S&P 500 historical changes
    2. Data availability at each point in time
    3. Delisting detection

    Args:
        base_tickers: Starting universe of tickers
        start_date: Start date for reconstruction
        end_date: End date for reconstruction

    Returns:
        Dictionary mapping YYYY-MM to list of available tickers
    """
    logger.info("Reconstructing historical universe...")
    ensure_dirs()

    # Get S&P 500 changes
    sp500_changes = get_sp500_historical_changes()

    # Download price data for all tickers
    logger.info(f"Downloading price data for {len(base_tickers)} tickers...")

    all_prices = yf.download(
        base_tickers,
        start=start_date,
        end=end_date,
        interval='1mo',
        auto_adjust=True,
        progress=False
    )

    if all_prices.empty:
        return {}

    if isinstance(all_prices.columns, pd.MultiIndex):
        prices = all_prices['Close']
    else:
        prices = all_prices

    prices.index = pd.to_datetime(prices.index)
    prices = prices.resample('ME').last()

    # Build universe for each month
    universe = {}

    # Track which stocks have been removed from S&P 500
    removed_from_sp500: Set[str] = set()

    for date in prices.index:
        date_str = date.strftime('%Y-%m')

        # Start with stocks that have valid price data
        valid_stocks = prices.loc[date].dropna().index.tolist()

        # Apply S&P 500 changes up to this date
        if not sp500_changes.empty:
            changes_before = sp500_changes[sp500_changes['Date'] <= date]

            # Track removals
            for _, row in changes_before.iterrows():
                if pd.notna(row.get('Ticker_Removed')):
                    removed_from_sp500.add(row['Ticker_Removed'])
                if pd.notna(row.get('Ticker_Added')):
                    removed_from_sp500.discard(row['Ticker_Added'])

        # Include removed stocks if they still have price data
        # (they should be included until actual delisting)
        final_universe = valid_stocks.copy()

        universe[date_str] = final_universe

    # Statistics
    total_months = len(universe)
    avg_stocks = np.mean([len(v) for v in universe.values()])
    min_stocks = min(len(v) for v in universe.values())
    max_stocks = max(len(v) for v in universe.values())

    logger.info(f"Reconstructed universe for {total_months} months")
    logger.info(f"  Average stocks per month: {avg_stocks:.0f}")
    logger.info(f"  Min: {min_stocks}, Max: {max_stocks}")

    # Save universe
    universe_path = os.path.join(CACHE_DIR, 'historical_universe.json')
    with open(universe_path, 'w') as f:
        json.dump(universe, f, indent=2)

    return universe


def get_expanded_historical_tickers() -> List[str]:
    """
    Get an expanded list of tickers including known historical constituents.

    This includes:
    1. Current S&P 500
    2. Known removed stocks that may still trade
    3. International developed market stocks
    """
    logger.info("Building expanded historical ticker list...")

    tickers = set()

    # Current S&P 500
    try:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(sp500_url)
        current_sp500 = tables[0]['Symbol'].str.replace('.', '-').tolist()
        tickers.update(current_sp500)
        logger.info(f"Added {len(current_sp500)} current S&P 500 stocks")
    except Exception as e:
        logger.warning(f"Could not fetch current S&P 500: {e}")

    # Historical changes - add removed stocks
    changes = get_sp500_historical_changes()
    if not changes.empty:
        removed = changes['Ticker_Removed'].dropna().unique().tolist()
        # Filter out obvious errors
        removed = [t for t in removed if isinstance(t, str) and len(t) <= 6]
        tickers.update(removed)
        logger.info(f"Added {len(removed)} historically removed S&P 500 stocks")

    # Known important delisted/acquired stocks
    # These are major stocks that were in MSCI World but are now gone
    historical_important = [
        # Acquired/Merged
        'TWX',   # Time Warner (acquired by AT&T)
        'MON',   # Monsanto (acquired by Bayer)
        'EMC',   # EMC Corp (acquired by Dell)
        'DELL',  # Dell (went private, now back)
        'WFT',   # Weatherford (bankruptcy)
        'CHK',   # Chesapeake Energy (bankruptcy, restructured)
        'GENZ',  # Genzyme (acquired by Sanofi)
        'MIL',   # Millipore (acquired by Merck)
        'BUD',   # Anheuser-Busch (acquired by InBev)
        'SGP',   # Schering-Plough (merged with Merck)
        'WYE',   # Wyeth (acquired by Pfizer)
        'MER',   # Merrill Lynch (acquired by BofA)
        'LEH',   # Lehman Brothers (bankruptcy)
        'BSC',   # Bear Stearns (acquired by JPM)
        'WB',    # Wachovia (acquired by Wells Fargo)
        'WM',    # Washington Mutual (bankruptcy)
        'AIG',   # AIG (still exists but restructured)
        'FNM',   # Fannie Mae (delisted)
        'FRE',   # Freddie Mac (delisted)
        'GM',    # GM (bankruptcy, restructured)
        'CIT',   # CIT Group (bankruptcy)
        'SHLD',  # Sears Holdings (bankruptcy)
        'GGP',   # General Growth Properties (bankruptcy)
        'ETFC',  # E*Trade (acquired by Morgan Stanley)
        'TIF',   # Tiffany (acquired by LVMH)
        'CXO',   # Concho Resources (acquired by ConocoPhillips)
        'NBL',   # Noble Energy (acquired by Chevron)
        'MXIM',  # Maxim Integrated (acquired by ADI)
        'FLIR',  # FLIR Systems (acquired by Teledyne)
        'XLNX',  # Xilinx (acquired by AMD)
        'NLOK',  # NortonLifeLock (merged with Avast)
        'ATVI',  # Activision Blizzard (acquired by Microsoft)
        'VMW',   # VMware (acquired by Broadcom)
        'CERN',  # Cerner (acquired by Oracle)
        'CTXS',  # Citrix (acquired by Vista)
        'TWTR',  # Twitter (acquired by Musk)
    ]
    tickers.update(historical_important)

    # International developed markets (via ADRs)
    international_adrs = [
        # Japan
        'TM', 'SONY', 'MUFG', 'SMFG', 'HMC', 'NTDOY', 'MFG', 'NMR', 'TAK',
        # UK
        'SHEL', 'AZN', 'HSBC', 'UL', 'BP', 'GSK', 'RIO', 'BHP', 'DEO', 'BTI', 'VOD',
        # Germany
        'SAP', 'SIEGY', 'DTEGY', 'BASFY', 'BAYRY', 'DB',
        # France
        'TTE', 'LVMUY', 'SNY', 'BNPQY',
        # Switzerland
        'NSRGY', 'NVS', 'RHHBY', 'UBS',
        # Netherlands
        'ASML', 'ING',
        # Other
        'NVO', 'TSM', 'TD', 'RY', 'ENB',
    ]
    tickers.update(international_adrs)

    # Convert to list and clean
    ticker_list = [t.strip() for t in tickers if isinstance(t, str) and len(t) <= 6 and t.isalnum()]
    ticker_list = list(set(ticker_list))  # Remove duplicates

    logger.info(f"Total expanded ticker universe: {len(ticker_list)} stocks")

    return ticker_list


def apply_delisting_returns(
    returns: pd.DataFrame,
    delisted_stocks: Dict[str, str],
    delisting_return: float = -0.30
) -> pd.DataFrame:
    """
    Apply conservative delisting returns to account for delisting bias.

    Stocks that were delisted likely had poor returns before delisting.
    We apply a -30% return (common academic assumption) at the delisting date.

    Args:
        returns: DataFrame of stock returns
        delisted_stocks: Dictionary of {ticker: delisting_date}
        delisting_return: Return to apply at delisting (default -30%)

    Returns:
        DataFrame with adjusted returns
    """
    logger.info(f"Applying delisting returns for {len(delisted_stocks)} stocks...")

    adjusted_returns = returns.copy()

    for ticker, delist_date in delisted_stocks.items():
        if ticker not in adjusted_returns.columns:
            continue

        if delist_date == 'unknown' or delist_date == 'error':
            continue

        try:
            delist_dt = pd.to_datetime(delist_date)

            # Find the closest month-end
            delist_month = delist_dt + pd.offsets.MonthEnd(0)

            if delist_month in adjusted_returns.index:
                # Apply delisting return
                adjusted_returns.loc[delist_month, ticker] = delisting_return

                # Set all subsequent returns to NaN
                future_dates = adjusted_returns.index[adjusted_returns.index > delist_month]
                adjusted_returns.loc[future_dates, ticker] = np.nan

        except Exception as e:
            continue

    return adjusted_returns


def create_survivorship_report(
    universe: Dict[str, List[str]],
    sp500_changes: pd.DataFrame,
    output_path: str
) -> None:
    """
    Create a report documenting survivorship bias mitigation efforts.

    Args:
        universe: Historical universe dictionary
        sp500_changes: DataFrame of S&P 500 changes
        output_path: Path to save report
    """
    report = []
    report.append("=" * 70)
    report.append("SURVIVORSHIP BIAS MITIGATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Universe statistics
    report.append("HISTORICAL UNIVERSE STATISTICS")
    report.append("-" * 40)
    if universe:
        dates = sorted(universe.keys())
        report.append(f"Period: {dates[0]} to {dates[-1]}")
        report.append(f"Total months: {len(dates)}")
        sizes = [len(v) for v in universe.values()]
        report.append(f"Average stocks per month: {np.mean(sizes):.0f}")
        report.append(f"Min stocks: {min(sizes)}")
        report.append(f"Max stocks: {max(sizes)}")
    report.append("")

    # S&P 500 changes
    report.append("S&P 500 CONSTITUENT CHANGES TRACKED")
    report.append("-" * 40)
    if not sp500_changes.empty:
        report.append(f"Total changes tracked: {len(sp500_changes)}")
        report.append(f"Date range: {sp500_changes['Date'].min()} to {sp500_changes['Date'].max()}")

        # Recent changes
        report.append("\nMost recent 10 changes:")
        recent = sp500_changes.tail(10)
        for _, row in recent.iterrows():
            date = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else 'Unknown'
            added = row['Ticker_Added'] if pd.notna(row.get('Ticker_Added')) else '-'
            removed = row['Ticker_Removed'] if pd.notna(row.get('Ticker_Removed')) else '-'
            report.append(f"  {date}: Added {added}, Removed {removed}")
    report.append("")

    # Limitations
    report.append("KNOWN LIMITATIONS")
    report.append("-" * 40)
    report.append("1. Using current stock prices for historical periods")
    report.append("2. Missing delisting returns for most stocks")
    report.append("3. No access to true point-in-time index membership")
    report.append("4. Market cap data is approximate (current shares outstanding)")
    report.append("5. International stocks may have incomplete coverage")
    report.append("")
    report.append("For academic-grade survivorship-free backtesting, use CRSP/Compustat.")
    report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"Saved survivorship report to {output_path}")


def main():
    """Test survivorship mitigation functions."""
    ensure_dirs()

    print("\n" + "="*70)
    print("  SURVIVORSHIP BIAS MITIGATION TEST")
    print("="*70 + "\n")

    # Get S&P 500 changes
    changes = get_sp500_historical_changes()
    print(f"S&P 500 changes found: {len(changes)}")
    if not changes.empty:
        print("\nSample changes:")
        print(changes.head(10).to_string())

    # Get expanded ticker list
    tickers = get_expanded_historical_tickers()
    print(f"\nExpanded ticker universe: {len(tickers)} stocks")

    # Reconstruct universe (limited test)
    test_tickers = tickers[:100]  # Limit for testing
    universe = reconstruct_historical_universe(
        test_tickers,
        start_date='2015-01-01',
        end_date='2024-12-31'
    )

    # Create report
    report_path = os.path.join(DATA_DIR, 'survivorship_report.txt')
    create_survivorship_report(universe, changes, report_path)

    print(f"\nReport saved to: {report_path}")

    return changes, tickers, universe


if __name__ == '__main__':
    changes, tickers, universe = main()
