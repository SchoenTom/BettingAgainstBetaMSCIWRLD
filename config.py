"""
config.py - Centralized configuration for BAB (Betting Against Beta) strategy

This module contains all configurable parameters used across the project.

Key Design Decisions:
- Beta estimation starts from 1995 (60-month rolling window)
- BAB portfolio construction from 2000 (first valid betas)
- End date: 2014 (before arbitrage profits disappear per academic literature)
- Uses S&P 500 (^GSPC) as market benchmark (full history available)
- 60-month rolling window for beta estimation (standard in literature)

Author: BAB Strategy Implementation
"""

import os
from datetime import datetime

# ============================================================================
# Directory Configuration
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

# ============================================================================
# Date Range Configuration
# ============================================================================
# Beta estimation requires 60 months of data -> first valid beta at 2000-01
# Analysis ends at 2014 (BAB arbitrage profits decline after this per literature)
#
# START_DATE: When to start downloading price data (1995 for beta estimation)
# END_DATE: End of analysis period (2014)
START_DATE = '1995-01-01'  # Start of beta estimation period
END_DATE = '2014-12-31'    # End of analysis (before arbitrage profits disappear)

# ============================================================================
# Beta Calculation Configuration
# ============================================================================
ROLLING_WINDOW = 60        # Rolling window for beta calculation (months)
MIN_PERIODS_BETA = 36      # Minimum periods required (allows partial window if needed)

# ============================================================================
# Portfolio Construction Configuration
# ============================================================================
NUM_QUINTILES = 5          # Number of beta groups (quintiles: Q1=low, Q5=high)
MIN_STOCKS_PER_QUINTILE = 5  # Minimum stocks needed per quintile

# ============================================================================
# Benchmark Configuration
# ============================================================================
BENCHMARK_TICKER = '^GSPC'  # S&P 500 as market proxy (full history from 1950s)
RISK_FREE_TICKER = '^IRX'   # 3-month T-bill rate

# ============================================================================
# Data Download Configuration
# ============================================================================
DOWNLOAD_BATCH_SIZE = 50   # Number of tickers per batch download
DOWNLOAD_INTERVAL = '1mo'  # Monthly data

# ============================================================================
# Backtest Configuration
# ============================================================================
PERIODS_PER_YEAR = 12      # Monthly data
ROLLING_SHARPE_WINDOW = 12 # Window for rolling Sharpe ratio

# ============================================================================
# Visualization Configuration
# ============================================================================
FIGURE_SIZE = (12, 6)
FIGURE_DPI = 150
COLORS = {
    'bab': '#1f77b4',       # Blue
    'benchmark': '#ff7f0e',  # Orange
    'q1': '#2ca02c',        # Green (low beta)
    'q5': '#d62728',        # Red (high beta)
    'spread': '#9467bd',    # Purple
    'neutral': '#7f7f7f',   # Gray
}

# ============================================================================
# MSCI World Constituents - Curated List
# Representative sample of major developed market stocks
# Focus on stocks that existed BEFORE 1995 for full data availability
# ============================================================================
MSCI_WORLD_TICKERS = [
    # United States - Large Cap (all existed before 1995)
    'AAPL', 'MSFT', 'JPM', 'JNJ', 'PG', 'XOM', 'HD',
    'CVX', 'MRK', 'PEP', 'KO', 'BAC', 'MCD', 'WMT',
    'PFE', 'ABT', 'NKE', 'TXN',
    'ORCL', 'UNP', 'LOW', 'INTC',
    'BMY', 'AMGN', 'T', 'HON', 'IBM',
    'CAT', 'BA', 'MMM', 'DE', 'AXP', 'MS',
    'GE', 'C', 'MO', 'BDX',
    'ETN', 'TJX', 'DUK', 'SO', 'WM',
    'LLY', 'DOW', 'EMR', 'FDX', 'GD', 'GIS',
    'HAL', 'HES', 'HPQ', 'ITW', 'K', 'KMB',
    'LMT', 'MDT', 'MET', 'NSC', 'PH', 'PNC',
    'PRU', 'SLB', 'TGT', 'TRV', 'USB', 'WFC', 'WY',
    'XRX', 'YUM', 'ZBH', 'AFL', 'AIG', 'ALL', 'AON',
    'AVY', 'AEP', 'D', 'ED', 'EXC', 'F', 'GM',
    'GLW', 'HIG', 'IP', 'KEY', 'L', 'LNC', 'MMC',
    'NEM', 'NUE', 'OXY', 'PEG', 'PPG', 'PPL', 'ROK',
    'SHW', 'STT', 'SWK', 'SYY', 'TEL', 'VLO', 'WHR',

    # Japan (ADRs - established before 1995)
    'TM', 'SONY', 'HMC',

    # United Kingdom (established before 1995)
    'BP', 'HSBC', 'UL', 'BHP', 'BTI', 'VOD',

    # Germany
    'SAP', 'DB',

    # France
    'SNY',

    # Switzerland
    'NVS', 'UBS',

    # Canada (established before 1995)
    'TD', 'RY', 'BNS', 'BMO', 'CM', 'BCE',
]


def ensure_directories():
    """Create data and output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_date_info():
    """Return formatted date information for logging."""
    return {
        'start': START_DATE,
        'end': END_DATE,
        'first_beta_date': '2000-01-01',  # 60 months after START_DATE (1995)
        'note': 'Beta estimation: 1995-2000, BAB portfolios: 2000-2014'
    }


if __name__ == '__main__':
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Rolling Window: {ROLLING_WINDOW} months")
    print(f"Number of Tickers: {len(MSCI_WORLD_TICKERS)}")
    print(f"First valid beta date: 2000-01 (60 months after 1995)")
