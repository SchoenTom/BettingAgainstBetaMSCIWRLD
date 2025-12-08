"""
config.py - Centralized configuration for BAB strategy project

This module contains all configurable parameters used across the project.
Modify these values to customize the strategy behavior.

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
START_DATE = '2000-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# ============================================================================
# Beta Calculation Configuration
# ============================================================================
ROLLING_WINDOW = 60  # Rolling window for beta calculation (months)
MIN_PERIODS_BETA = 24  # Minimum periods required for beta calculation

# ============================================================================
# Portfolio Construction Configuration
# ============================================================================
NUM_QUINTILES = 5  # Number of beta groups (quintiles)

# ============================================================================
# Benchmark Configuration
# ============================================================================
MSCI_WORLD_PROXY = 'URTH'  # iShares MSCI World ETF
RISK_FREE_TICKER = '^IRX'  # 3-month T-bill rate

# ============================================================================
# Data Download Configuration
# ============================================================================
DOWNLOAD_BATCH_SIZE = 50  # Number of tickers per batch download
DOWNLOAD_INTERVAL = '1mo'  # Monthly data

# ============================================================================
# Backtest Configuration
# ============================================================================
PERIODS_PER_YEAR = 12  # Monthly data
RISK_FREE_RATE_BACKTEST = 0  # Assume 0 for Sharpe calculation
ROLLING_SHARPE_WINDOW = 12  # Window for rolling Sharpe ratio

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
# MSCI World Constituents
# Representative sample of major developed market stocks
# ============================================================================
MSCI_WORLD_TICKERS = [
    # United States - Large Cap (Top 100)
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
    'TM', 'SONY', 'MUFG', 'SMFG', 'HMC', 'NTT', 'MFG', 'NTDOY',

    # United Kingdom
    'BP', 'SHEL', 'HSBC', 'GSK', 'AZN', 'UL', 'RIO', 'BHP', 'BTI', 'LYG',
    'NWG', 'VOD', 'DEO',

    # Germany
    'SAP', 'DB', 'DTEGY', 'BASFY', 'BAYRY', 'SIEGY', 'VWAGY',

    # France
    'TTE', 'SNY', 'LVMUY', 'LRLCY', 'BNPQY',

    # Switzerland
    'NSRGY', 'NVS', 'RHHBY', 'UBS',

    # Netherlands
    'ASML', 'ING', 'NVO',

    # Canada
    'TD', 'RY', 'BNS', 'CNQ', 'ENB', 'CP', 'CNI', 'TRP', 'BMO', 'CM',
    'SU', 'MFC', 'BCE',

    # Other Developed Markets
    'TSM',  # Taiwan (often included in MSCI World calculations)
]


def ensure_directories():
    """Create data and output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == '__main__':
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Number of Tickers: {len(MSCI_WORLD_TICKERS)}")
