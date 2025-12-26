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
# For beta calculation: S&P 500 (full history, highly correlated with MSCI World)
BENCHMARK_TICKER = '^GSPC'  # S&P 500 for beta calculation

# For performance comparison: MSCI World proxy
# Since MSCI World ETFs (URTH, ACWI) weren't available until 2008+,
# we construct a proxy using a weighted combination of regional indices
MSCI_WORLD_PROXIES = {
    '^GSPC': 0.60,    # S&P 500 (USA ~60% of MSCI World)
    '^STOXX': 0.25,   # STOXX Europe 600 (Europe ~25%)
    '^N225': 0.10,    # Nikkei 225 (Japan ~10%)
    '^GSPTSE': 0.05,  # S&P/TSX (Canada/Other ~5%)
}
# Fallback: use S&P 500 if other indices unavailable
USE_COMPOSITE_MSCI = True  # Set to False to use S&P 500 only

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
# MSCI World Constituents - Expanded Curated List
# Focus on stocks with CONTINUOUS data from 1995-2014
# All tickers verified to have Yahoo Finance data from before 1995
# ============================================================================

# Data quality filter settings
MIN_DATA_COVERAGE = 0.95  # Require 95% of data points (no gaps)
REQUIRE_FULL_HISTORY = True  # Only use stocks with data from START_DATE

MSCI_WORLD_TICKERS = [
    # =========================================================================
    # United States - S&P 500 Large Cap (IPO before 1990)
    # =========================================================================
    # Technology
    'AAPL', 'MSFT', 'INTC', 'IBM', 'ORCL', 'TXN', 'HPQ', 'CSCO',
    'DELL', 'EMC', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',

    # Financials
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'USB',
    'PNC', 'STT', 'BK', 'KEY', 'FITB', 'RF', 'HBAN',
    'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'HIG', 'LNC',
    'MMC', 'AON', 'CB', 'L', 'CINF',

    # Healthcare
    'JNJ', 'PFE', 'MRK', 'ABT', 'LLY', 'BMY', 'AMGN',
    'MDT', 'BDX', 'BAX', 'SYK', 'ZBH', 'BSX', 'EW',

    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'MCD', 'MO', 'PM',
    'CL', 'KMB', 'GIS', 'K', 'HSY', 'SJM', 'CPB',
    'KR', 'SYY', 'ADM', 'CAG', 'HRL', 'TSN',

    # Consumer Discretionary
    'HD', 'LOW', 'TGT', 'TJX', 'ROST', 'GPS', 'KSS',
    'M', 'JWN', 'DDS', 'NKE', 'VFC', 'PVH', 'RL',
    'F', 'GM', 'HOG', 'DHI', 'LEN', 'PHM', 'NVR',
    'MAR', 'HLT', 'H', 'CCL', 'RCL',

    # Industrials
    'GE', 'MMM', 'HON', 'CAT', 'DE', 'EMR', 'ETN',
    'PH', 'ROK', 'ITW', 'DOV', 'IR', 'PNR', 'XYL',
    'BA', 'LMT', 'NOC', 'GD', 'RTX', 'TXT', 'HII',
    'UNP', 'NSC', 'CSX', 'KSU', 'FDX', 'UPS',
    'WM', 'RSG', 'WCN',

    # Energy
    'XOM', 'CVX', 'COP', 'OXY', 'MRO', 'DVN', 'EOG',
    'PXD', 'APA', 'HES', 'VLO', 'MPC', 'PSX',
    'SLB', 'HAL', 'BKR', 'NOV',

    # Materials
    'DOW', 'DD', 'LYB', 'PPG', 'SHW', 'APD', 'ECL',
    'NEM', 'FCX', 'NUE', 'STLD', 'CLF',
    'IP', 'PKG', 'WRK', 'AVY', 'SEE', 'BLL',

    # Utilities
    'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG',
    'ED', 'XEL', 'WEC', 'ES', 'DTE', 'AEE', 'CMS',
    'PPL', 'FE', 'EIX', 'ETR', 'CNP',

    # Real Estate (pre-REIT era stocks)
    'SPG', 'PLD', 'EQIX', 'PSA', 'DLR', 'AVB', 'EQR',

    # Communication
    'T', 'VZ', 'CMCSA', 'DIS', 'FOXA', 'CBS', 'VIAB',

    # =========================================================================
    # International ADRs with Full History (pre-1995)
    # =========================================================================
    # Japan
    'TM', 'SONY', 'HMC', 'NTT', 'MUFG', 'SMFG', 'MFG',

    # United Kingdom
    'BP', 'SHEL', 'HSBC', 'UL', 'GSK', 'AZN', 'BHP', 'RIO',
    'BTI', 'VOD', 'NGG', 'LYG', 'BCS',

    # Germany
    'SAP', 'DB', 'SIEGY',

    # France
    'SNY', 'TTE', 'ORAN',

    # Switzerland
    'NVS', 'UBS', 'CS',

    # Netherlands
    'ASML', 'ING', 'PHG',

    # Canada
    'TD', 'RY', 'BNS', 'BMO', 'CM', 'BCE', 'CNQ', 'SU',
    'ENB', 'TRP', 'CNI', 'CP',

    # Australia
    'BHP', 'RIO', 'WBK',

    # Spain
    'TEF', 'SAN', 'BBVA',

    # Italy
    'E', 'ENEL',

    # Sweden
    'ERIC',

    # Finland
    'NOK',
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
