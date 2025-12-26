"""
config.py - Centralized configuration for BAB (Betting Against Beta) strategy

This module contains all configurable parameters used across the project.

Key Design Decisions:
- Date range starts from 2012 (URTH inception) to allow 60-month beta calculation
- First valid betas available from 2017 (after 60 months of URTH data)
- Uses URTH as MSCI World proxy (iShares MSCI World ETF)
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
# IMPORTANT: URTH (iShares MSCI World ETF) started trading in January 2012.
# We need 60 months of data to compute the first valid rolling beta.
# Therefore: First valid beta = January 2017
#
# START_DATE: When to start downloading price data (need 5 years before first beta)
# END_DATE: End of analysis period
START_DATE = '2012-01-01'  # URTH inception - need full history for beta calculation
END_DATE = '2024-12-31'    # Current analysis end

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
MSCI_WORLD_PROXY = 'URTH'  # iShares MSCI World ETF (inception: Jan 2012)
RISK_FREE_TICKER = '^IRX'  # 3-month T-bill rate

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
# Representative sample of major developed market stocks with long histories
# Focus on stocks that existed before 2012 to maximize data availability
# ============================================================================
MSCI_WORLD_TICKERS = [
    # United States - Large Cap (established companies with long histories)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'HD',
    'CVX', 'MA', 'ABBV', 'MRK', 'PEP', 'KO', 'BAC', 'MCD', 'CSCO', 'WMT',
    'TMO', 'PFE', 'ABT', 'CRM', 'ACN', 'ADBE', 'DHR', 'NKE', 'TXN', 'NFLX',
    'ORCL', 'UNP', 'PM', 'RTX', 'NEE', 'LOW', 'QCOM', 'INTC', 'LIN',
    'BMY', 'CMCSA', 'AMGN', 'T', 'VZ', 'UPS', 'HON', 'COP', 'SBUX', 'IBM',
    'GS', 'CAT', 'BA', 'MMM', 'DE', 'BLK', 'AXP', 'GILD', 'MS', 'CVS',
    'MDLZ', 'GE', 'C', 'ADI', 'PLD', 'AMT', 'ISRG',
    'ZTS', 'SYK', 'BKNG', 'ADP', 'MMC', 'SPGI', 'MO', 'CB', 'BDX',
    'ETN', 'TJX', 'DUK', 'SO', 'CME', 'PGR', 'WM', 'CI',
    'UNH', 'LLY', 'COST', 'AVGO', 'AMD',

    # Japan (ADRs with long histories)
    'TM', 'SONY', 'MUFG', 'SMFG', 'HMC', 'NTT', 'MFG',

    # United Kingdom
    'BP', 'SHEL', 'HSBC', 'GSK', 'AZN', 'UL', 'RIO', 'BHP', 'BTI',
    'VOD', 'DEO',

    # Germany
    'SAP', 'DB', 'SIEGY',

    # France
    'TTE', 'SNY',

    # Switzerland
    'NSRGY', 'NVS', 'UBS',

    # Netherlands
    'ASML', 'ING',

    # Canada
    'TD', 'RY', 'BNS', 'CNQ', 'ENB', 'CP', 'CNI', 'TRP', 'BMO', 'CM',
    'SU', 'MFC', 'BCE',

    # Taiwan
    'TSM',
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
        'first_beta_date': '2017-01-01',  # 60 months after START_DATE
        'note': 'URTH inception: 2012-01, first valid beta: 2017-01'
    }


if __name__ == '__main__':
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Rolling Window: {ROLLING_WINDOW} months")
    print(f"Number of Tickers: {len(MSCI_WORLD_TICKERS)}")
    print(f"First valid beta date: ~2017-01 (60 months after URTH inception)")
