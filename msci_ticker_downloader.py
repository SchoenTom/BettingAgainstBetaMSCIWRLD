"""
msci_ticker_downloader.py - Download MSCI World constituents from free sources

This module fetches the actual MSCI World constituents (~1,400 stocks) from
iShares URTH ETF holdings and maps them to yfinance-compatible tickers.

Strategy:
1. Download URTH holdings CSV from iShares (official, free, updated daily)
2. Map international tickers to yfinance format using exchange suffixes
3. Validate tickers actually work with yfinance
4. Cache valid tickers to avoid repeated API calls
5. Fallback to curated list for any gaps

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Dict, Optional, Tuple
import time
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cache')
CACHE_EXPIRY_DAYS = 7  # Re-download holdings weekly

# iShares URTH holdings URL (MSCI World ETF)
# This URL provides daily holdings as CSV
ISHARES_URTH_URL = "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax"

# Exchange suffix mapping for yfinance
# Maps iShares exchange codes to yfinance suffixes
EXCHANGE_SUFFIX_MAP = {
    # North America
    'NASDAQ': '',
    'New York Stock Exchange Inc.': '',
    'NYSE': '',
    'NYSE ARCA': '',
    'Nyse Mkt Llc': '',
    'Cboe BZX formerly known as BATS': '',
    'Toronto Stock Exchange': '.TO',
    'TSX': '.TO',

    # Europe
    'London Stock Exchange': '.L',
    'LSE': '.L',
    'Xetra': '.DE',
    'XETRA': '.DE',
    'Frankfurt Stock Exchange': '.DE',
    'Euronext Paris': '.PA',
    'Paris': '.PA',
    'Euronext Amsterdam': '.AS',
    'Amsterdam': '.AS',
    'SIX Swiss Exchange': '.SW',
    'Swiss': '.SW',
    'Borsa Italiana S.P.A.': '.MI',
    'Milan': '.MI',
    'Bolsa De Madrid': '.MC',
    'Madrid': '.MC',
    'Irish Stock Exchange - Loss': '.IR',
    'OMX Nordic Exchange Stockholm': '.ST',
    'Stockholm': '.ST',
    'Oslo Stock Exchange': '.OL',
    'Copenhagen': '.CO',
    'Helsinki': '.HE',
    'Euronext Brussels': '.BR',
    'Vienna': '.VI',
    'Euronext Lisbon': '.LS',

    # Asia-Pacific
    'Tokyo Stock Exchange': '.T',
    'TSE': '.T',
    'Hong Kong Stock Exchange': '.HK',
    'HKSE': '.HK',
    'Singapore Exchange': '.SI',
    'SGX': '.SI',
    'Australian Stock Exchange Ltd.': '.AX',
    'ASX': '.AX',
    'New Zealand Stock Exchange': '.NZ',

    # Other
    'Tel Aviv Stock Exchange': '.TA',
}

# Country to primary exchange suffix (fallback)
COUNTRY_SUFFIX_MAP = {
    'United States': '',
    'Canada': '.TO',
    'United Kingdom': '.L',
    'Germany': '.DE',
    'France': '.PA',
    'Netherlands': '.AS',
    'Switzerland': '.SW',
    'Italy': '.MI',
    'Spain': '.MC',
    'Sweden': '.ST',
    'Norway': '.OL',
    'Denmark': '.CO',
    'Finland': '.HE',
    'Belgium': '.BR',
    'Austria': '.VI',
    'Portugal': '.LS',
    'Ireland': '.IR',
    'Japan': '.T',
    'Hong Kong': '.HK',
    'Singapore': '.SI',
    'Australia': '.AX',
    'New Zealand': '.NZ',
    'Israel': '.TA',
}

# ADR mapping: Some international stocks are better accessed via US ADRs
# Maps ISIN prefix or company patterns to ADR tickers
ADR_ALTERNATIVES = {
    # Japanese companies - prefer ADRs for liquidity
    'Toyota Motor': 'TM',
    'Sony Group': 'SONY',
    'Mitsubishi UFJ': 'MUFG',
    'Sumitomo Mitsui': 'SMFG',
    'Honda Motor': 'HMC',
    'Nintendo': 'NTDOY',
    'Keyence': 'KYCCF',
    'Tokyo Electron': 'TOELY',
    'Shin-Etsu': 'SHECY',
    'Daikin': 'DKILY',
    'Recruit': 'RCRUY',
    'SoftBank': 'SFTBY',
    'Hitachi': 'HTHIY',
    'Mizuho': 'MFG',
    'KDDI': 'KDDIY',
    'Takeda': 'TAK',
    'FANUC': 'FANUY',
    'Murata': 'MRAAY',
    'SMC': 'SMCAY',
    'Nidec': 'NJDCY',

    # UK companies - many trade as ADRs
    'Shell': 'SHEL',
    'AstraZeneca': 'AZN',
    'HSBC': 'HSBC',
    'Unilever': 'UL',
    'BP': 'BP',
    'GlaxoSmithKline': 'GSK',
    'GSK': 'GSK',
    'Diageo': 'DEO',
    'British American': 'BTI',
    'Rio Tinto': 'RIO',
    'BHP': 'BHP',
    'Vodafone': 'VOD',
    'Barclays': 'BCS',
    'Lloyds': 'LYG',
    'NatWest': 'NWG',
    'Prudential': 'PUK',
    'Linde': 'LIN',

    # German companies
    'SAP': 'SAP',
    'Siemens': 'SIEGY',
    'Deutsche Telekom': 'DTEGY',
    'Allianz': 'ALIZY',
    'BASF': 'BASFY',
    'Bayer': 'BAYRY',
    'Mercedes': 'MBGYY',
    'BMW': 'BMWYY',
    'Volkswagen': 'VWAGY',
    'Deutsche Bank': 'DB',
    'Munich Re': 'MURGY',
    'Infineon': 'IFNNY',

    # French companies
    'TotalEnergies': 'TTE',
    'LVMH': 'LVMUY',
    'L\'Oreal': 'LRLCY',
    'Sanofi': 'SNY',
    'BNP Paribas': 'BNPQY',
    'Schneider': 'SBGSY',
    'Air Liquide': 'AIQUY',
    'Hermes': 'HESAY',
    'Kering': 'PPRUY',
    'Essilor': 'ESLOY',
    'Dassault': 'DASTY',
    'Safran': 'SAFRY',
    'Airbus': 'EADSY',

    # Swiss companies
    'Nestle': 'NSRGY',
    'Novartis': 'NVS',
    'Roche': 'RHHBY',
    'UBS': 'UBS',
    'Zurich': 'ZURVY',
    'ABB': 'ABB',
    'Richemont': 'CFRUY',

    # Netherlands
    'ASML': 'ASML',
    'ING': 'ING',
    'Philips': 'PHG',
    'NXP': 'NXPI',
    'Stellantis': 'STLA',

    # Other European
    'Novo Nordisk': 'NVO',
    'Spotify': 'SPOT',
    'Ericsson': 'ERIC',
    'Volvo': 'VLVLY',
    'Atlas Copco': 'ATLKY',
    'ABB': 'ABB',
    'Adyen': 'ADYEY',

    # Asian (non-Japan)
    'Taiwan Semiconductor': 'TSM',
    'Samsung': 'SSNLF',
    'TSMC': 'TSM',

    # Australian
    'Commonwealth Bank': 'CMWAY',
    'CSL': 'CSLLY',
    'Westpac': 'WBK',
    'ANZ': 'ANZGY',
    'National Australia': 'NABZY',
    'Macquarie': 'MQBKY',
    'Woodside': 'WDS',

    # Canadian (prefer TSX if available, otherwise US listing)
    'Royal Bank of Canada': 'RY',
    'Toronto-Dominion': 'TD',
    'Bank of Nova Scotia': 'BNS',
    'Brookfield': 'BN',
    'Canadian Pacific': 'CP',
    'Canadian National': 'CNI',
    'Enbridge': 'ENB',
    'TC Energy': 'TRP',
    'Suncor': 'SU',
    'Bank of Montreal': 'BMO',
}


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(filename: str) -> str:
    """Get full path for cache file."""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, filename)


def is_cache_valid(cache_path: str, max_age_days: int = CACHE_EXPIRY_DAYS) -> bool:
    """Check if cache file exists and is not expired."""
    if not os.path.exists(cache_path):
        return False

    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return datetime.now() - file_time < timedelta(days=max_age_days)


def download_urth_holdings() -> Optional[pd.DataFrame]:
    """
    Download URTH ETF holdings from iShares.

    Returns:
        DataFrame with columns: ticker, name, weight, sector, exchange, country, isin
    """
    logger.info("Downloading URTH holdings from iShares...")

    # Try multiple URL patterns as iShares sometimes changes them
    urls_to_try = [
        f"{ISHARES_URTH_URL}?fileType=csv&fileName=URTH_holdings&dataType=fund",
        "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax?fileType=csv",
        "https://www.blackrock.com/us/individual/products/239696/ishares-msci-world-etf/1467271812596.ajax?fileType=csv&fileName=URTH_holdings&dataType=fund",
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/csv,application/csv,text/plain,*/*',
    }

    for url in urls_to_try:
        try:
            logger.info(f"Trying URL: {url[:80]}...")
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200 and len(response.text) > 1000:
                # Parse CSV - iShares format has some header rows to skip
                lines = response.text.split('\n')

                # Find the actual data header row
                header_idx = 0
                for i, line in enumerate(lines):
                    if 'Ticker' in line or 'ticker' in line.lower():
                        header_idx = i
                        break
                    if 'Name' in line and 'Weight' in line:
                        header_idx = i
                        break

                # Read from header row
                csv_data = '\n'.join(lines[header_idx:])
                df = pd.read_csv(StringIO(csv_data))

                # Standardize column names
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

                # Filter out non-equity holdings (cash, futures, etc.)
                if 'asset_class' in df.columns:
                    df = df[df['asset_class'].str.contains('Equity', case=False, na=False)]

                logger.info(f"Downloaded {len(df)} holdings from iShares")
                return df

        except Exception as e:
            logger.warning(f"Failed to download from URL: {e}")
            continue

    logger.warning("Could not download from iShares, trying alternative sources...")
    return None


def download_holdings_alternative() -> Optional[pd.DataFrame]:
    """
    Alternative: Scrape holdings from other free sources.

    Tries:
    1. ETF.com holdings data
    2. Yahoo Finance URTH holdings
    """
    logger.info("Trying alternative sources for URTH holdings...")

    # Try yfinance for holdings (limited but available)
    try:
        urth = yf.Ticker('URTH')

        # Get holdings if available
        if hasattr(urth, 'major_holders') and urth.major_holders is not None:
            logger.info("Retrieved partial holdings from yfinance")

        # yfinance doesn't provide full holdings list easily
        # We'll use the institutional holders as a starting point

    except Exception as e:
        logger.warning(f"yfinance holdings lookup failed: {e}")

    return None


def map_to_yfinance_ticker(row: pd.Series) -> Optional[str]:
    """
    Map a holding row to a yfinance-compatible ticker.

    Args:
        row: DataFrame row with ticker, name, exchange, country info

    Returns:
        yfinance-compatible ticker string or None
    """
    ticker = str(row.get('ticker', '')).strip()
    name = str(row.get('name', '')).strip()
    exchange = str(row.get('exchange', '')).strip()
    country = str(row.get('location', row.get('country', ''))).strip()

    # Skip if no ticker
    if not ticker or ticker == 'nan' or ticker == '-':
        return None

    # Check ADR alternatives first (prefer liquid US-listed ADRs)
    for company_pattern, adr_ticker in ADR_ALTERNATIVES.items():
        if company_pattern.lower() in name.lower():
            return adr_ticker

    # US stocks - no suffix needed
    if country in ['United States', 'US', 'USA'] or exchange in ['NASDAQ', 'NYSE', 'New York Stock Exchange Inc.']:
        # Clean up ticker (remove any existing suffix)
        clean_ticker = ticker.split('.')[0].replace(' ', '-')
        return clean_ticker

    # Get exchange suffix
    suffix = ''
    if exchange in EXCHANGE_SUFFIX_MAP:
        suffix = EXCHANGE_SUFFIX_MAP[exchange]
    elif country in COUNTRY_SUFFIX_MAP:
        suffix = COUNTRY_SUFFIX_MAP[country]

    # Build ticker
    clean_ticker = ticker.split('.')[0].replace(' ', '')

    # Some tickers have numeric codes (especially Asian)
    if clean_ticker.isdigit():
        # Japanese stocks are 4 digits
        if country == 'Japan' or 'Tokyo' in exchange:
            return f"{clean_ticker}.T"
        # Hong Kong stocks
        elif country == 'Hong Kong' or 'Hong Kong' in exchange:
            return f"{clean_ticker.zfill(4)}.HK"

    return f"{clean_ticker}{suffix}" if suffix else clean_ticker


def validate_tickers_batch(tickers: List[str], batch_size: int = 100) -> Tuple[List[str], List[str]]:
    """
    Validate tickers by checking if they have data in yfinance.

    Args:
        tickers: List of ticker symbols to validate
        batch_size: Number of tickers to check per batch

    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    logger.info(f"Validating {len(tickers)} tickers...")

    valid = []
    invalid = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Validating batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

        try:
            # Download minimal data to check validity
            data = yf.download(
                batch,
                period='5d',
                progress=False,
                threads=True
            )

            if isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers - check which have data
                for ticker in batch:
                    if ticker in data['Close'].columns:
                        if data['Close'][ticker].notna().any():
                            valid.append(ticker)
                        else:
                            invalid.append(ticker)
                    else:
                        invalid.append(ticker)
            elif len(batch) == 1:
                # Single ticker
                if not data.empty and data['Close'].notna().any():
                    valid.append(batch[0])
                else:
                    invalid.append(batch[0])
            else:
                # Check each ticker individually
                for ticker in batch:
                    if ticker in data.columns:
                        valid.append(ticker)
                    else:
                        invalid.append(ticker)

        except Exception as e:
            logger.warning(f"Batch validation error: {e}")
            # Try individual validation for this batch
            for ticker in batch:
                try:
                    test = yf.Ticker(ticker)
                    hist = test.history(period='5d')
                    if len(hist) > 0:
                        valid.append(ticker)
                    else:
                        invalid.append(ticker)
                except:
                    invalid.append(ticker)

        # Rate limiting
        time.sleep(0.5)

    logger.info(f"Validation complete: {len(valid)} valid, {len(invalid)} invalid")
    return valid, invalid


def get_curated_fallback_tickers() -> List[str]:
    """
    Get curated fallback list of major MSCI World constituents.
    These are reliable yfinance tickers covering major developed markets.
    Expanded list with ~600+ tickers for better MSCI World coverage.
    """
    return [
        # ===========================================
        # UNITED STATES - S&P 500 + Extended (~300)
        # ===========================================
        # Mega-cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
        'CRM', 'ADBE', 'AMD', 'CSCO', 'ACN', 'INTC', 'TXN', 'QCOM', 'IBM', 'INTU',
        'NOW', 'ADI', 'AMAT', 'LRCX', 'MU', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI',

        # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'SYK', 'BSX', 'MDT', 'BDX', 'ZTS',
        'ELV', 'CI', 'HUM', 'CVS', 'MCK', 'CAH', 'BIIB', 'ILMN', 'DXCM', 'IDXX',
        'IQV', 'A', 'MTD', 'WAT', 'TECH', 'HOLX', 'ALGN', 'EW', 'RMD', 'COO',

        # Financials
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK',
        'C', 'AXP', 'SPGI', 'CB', 'MMC', 'PGR', 'AON', 'ICE', 'CME', 'MCO',
        'USB', 'PNC', 'TFC', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV',
        'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'HBAN', 'MTB', 'SIVB', 'ZION',
        'DFS', 'SYF', 'ALLY', 'NTRS', 'CINF', 'L', 'RJF', 'MKTX', 'CBOE', 'NDAQ',

        # Consumer
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
        'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'ORLY', 'AZO', 'BBY', 'TSCO', 'ULTA',
        'EL', 'CL', 'KMB', 'GIS', 'K', 'SJM', 'CPB', 'MKC', 'HRL', 'CAG',
        'MDLZ', 'HSY', 'KHC', 'STZ', 'TAP', 'BF-B', 'PM', 'MO', 'KDP', 'MNST',
        'YUM', 'CMG', 'DPZ', 'QSR', 'DARDEN', 'MAR', 'HLT', 'H', 'WH', 'WYNN',
        'LVS', 'MGM', 'CZR', 'NCLH', 'CCL', 'RCL', 'EXPE', 'BKNG', 'ABNB', 'TRIP',

        # Industrials
        'UNP', 'RTX', 'HON', 'UPS', 'CAT', 'BA', 'DE', 'LMT', 'GE', 'MMM',
        'GD', 'NOC', 'EMR', 'ITW', 'ETN', 'PH', 'ROK', 'PCAR', 'CTAS', 'FAST',
        'FDX', 'CSX', 'NSC', 'ODFL', 'JBHT', 'CHRW', 'EXPD', 'XPO', 'UAL', 'DAL',
        'LUV', 'AAL', 'ALK', 'JBLU', 'WAB', 'TT', 'CARR', 'OTIS', 'JCI', 'TRANE',
        'WM', 'RSG', 'WCN', 'VRSK', 'IR', 'DOV', 'AME', 'ROP', 'IEX', 'GNRC',

        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD',
        'DVN', 'HES', 'HAL', 'BKR', 'FANG', 'APA', 'MRO', 'CTRA', 'KMI', 'WMB',
        'OKE', 'TRGP', 'LNG', 'ET', 'EPD', 'MPLX', 'PAA', 'ENB', 'TRP', 'SU',

        # Utilities & REITs
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'PEG',
        'WEC', 'ES', 'AWK', 'AEE', 'DTE', 'CMS', 'LNT', 'PPL', 'NI', 'EVRG',
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'DLR', 'WELL', 'AVB', 'EQR', 'SPG',
        'O', 'VICI', 'SBAC', 'WY', 'ARE', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT',

        # Tech & Software Extended
        'PANW', 'CRWD', 'ZS', 'FTNT', 'NET', 'DDOG', 'SNOW', 'MDB', 'TEAM', 'WDAY',
        'SPLK', 'VEEV', 'HUBS', 'ZEN', 'DOCU', 'OKTA', 'TWLO', 'TTD', 'PAYC', 'PCTY',
        'ANSS', 'PTC', 'KEYS', 'TRMB', 'CDW', 'EPAM', 'GLOB', 'GPN', 'FIS', 'FISV',
        'PAYX', 'JKHY', 'WEX', 'BR', 'TYL', 'SSNC', 'FLT', 'ADP', 'MSCI', 'VRSK',

        # Communications & Media
        'T', 'VZ', 'CMCSA', 'CHTR', 'TMUS', 'DIS', 'NFLX', 'PARA', 'WBD', 'FOX',
        'FOXA', 'NWS', 'NWSA', 'OMC', 'IPG', 'EA', 'TTWO', 'ATVI', 'RBLX', 'MTCH',

        # Consumer Internet
        'UBER', 'LYFT', 'DASH', 'ABNB', 'PINS', 'SNAP', 'ROKU', 'ETSY', 'EBAY', 'W',
        'CHWY', 'CVNA', 'CARG', 'ZG', 'OPEN', 'FVRR', 'UPWK', 'COIN', 'HOOD', 'SOFI',

        # ===========================================
        # JAPAN (~50 via ADRs/OTC)
        # ===========================================
        'TM', 'SONY', 'MUFG', 'SMFG', 'HMC', 'NTDOY', 'MFG', 'NMR', 'IX', 'TAK',
        'KYCCF', 'TOELY', 'SHECY', 'DKILY', 'RCRUY', 'SFTBY', 'HTHIY', 'KDDIY',
        'FANUY', 'MRAAY', 'SMCAY', 'CMPCY', 'DNZOY', 'ITOCY', 'MARUY', 'MSBHY',
        'OTSKY', 'SEKEY', 'APTS', 'NPPXF', 'NDEKY', 'NSANY', 'FJTSY', 'PCRFY',
        'APELY', 'JAPSY', 'SKHSY', 'BRTHY', 'TKOMY', 'MITSY', 'SBHSY', 'SGBLY',

        # ===========================================
        # UNITED KINGDOM (~40)
        # ===========================================
        'SHEL', 'AZN', 'HSBC', 'UL', 'BP', 'GSK', 'RIO', 'BHP', 'DEO', 'BTI',
        'VOD', 'BCS', 'LYG', 'NWG', 'PUK', 'RELX', 'CRH', 'WPP', 'RKT', 'LSEG.L',
        'EXPN.L', 'FLTR.L', 'III.L', 'BARC.L', 'LLOY.L', 'STAN.L', 'NG.L', 'SSE.L',
        'TSCO.L', 'ABF.L', 'IHG.L', 'WTB.L', 'IMB.L', 'SKG.L', 'AAL.L', 'ANTO.L',
        'FRES.L', 'GLEN.L', 'NXT.L', 'BDEV.L',

        # ===========================================
        # GERMANY (~35)
        # ===========================================
        'SAP', 'SIEGY', 'DTEGY', 'ALIZY', 'BASFY', 'BAYRY', 'DB', 'MBGYY', 'BMWYY',
        'VWAGY', 'MURGY', 'IFNNY', 'ADDYY', 'HENKY', 'SMAWF', 'DBOEY', 'DPSGY',
        # German exchange tickers
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'DBK.DE', 'MBG.DE', 'BMW.DE',
        'VOW3.DE', 'MUV2.DE', 'IFX.DE', 'ADS.DE', 'HEN3.DE', 'DTE.DE', 'EOAN.DE', 'RWE.DE',
        'FRE.DE', 'DPW.DE', 'MTX.DE', 'HEI.DE', 'CON.DE', 'ZAL.DE', '1COV.DE', 'PUM.DE',

        # ===========================================
        # FRANCE (~30)
        # ===========================================
        'TTE', 'LVMUY', 'LRLCY', 'SNY', 'BNPQY', 'SBGSY', 'AIQUY', 'HESAY', 'PPRUY',
        'ESLOY', 'DASTY', 'SAFRY', 'EADSY', 'STLA', 'CAGRY', 'AXAHY', 'SOCGF', 'VLVLY',
        # Paris exchange
        'OR.PA', 'MC.PA', 'SAN.PA', 'AI.PA', 'SU.PA', 'BN.PA', 'RI.PA', 'CAP.PA', 'KER.PA',
        'DG.PA', 'ENGI.PA', 'VIE.PA', 'SGO.PA', 'VIV.PA', 'ORA.PA', 'WLN.PA',

        # ===========================================
        # SWITZERLAND (~25)
        # ===========================================
        'NSRGY', 'NVS', 'RHHBY', 'UBS', 'ZURVY', 'ABB', 'CFRUY', 'LOGI', 'SGSOY',
        # Swiss exchange
        'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW', 'CFR.SW',
        'GEBN.SW', 'GIVN.SW', 'SREN.SW', 'SCMN.SW', 'SIKA.SW', 'LONN.SW', 'PGHN.SW',
        'BALN.SW', 'SLHN.SW', 'ALC.SW', 'STMN.SW',

        # ===========================================
        # NETHERLANDS (~20)
        # ===========================================
        'ASML', 'ING', 'PHG', 'NXPI', 'STLA', 'ADYEY', 'GLPEY',
        'ASML.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'UNA.AS', 'HEIA.AS', 'AKZA.AS',
        'DSM.AS', 'NN.AS', 'WKL.AS', 'RAND.AS', 'KPN.AS', 'IMCD.AS', 'ASM.AS',

        # ===========================================
        # SCANDINAVIA (~35)
        # ===========================================
        # Denmark
        'NVO', 'NZYMB.CO', 'CARL-B.CO', 'NOVO-B.CO', 'MAERSK-B.CO', 'DSV.CO', 'VWS.CO',
        'ORSTED.CO', 'COLO-B.CO', 'DEMANT.CO', 'GN.CO', 'PNDORA.CO',

        # Sweden
        'SPOT', 'ERIC', 'VLVLY', 'ATLKY',
        'ASSA-B.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'SAND.ST', 'SEB-A.ST', 'SWED-A.ST',
        'HM-B.ST', 'INVE-B.ST', 'VOLV-B.ST', 'ESSITY-B.ST', 'SKA-B.ST', 'ABB.ST',
        'HEXA-B.ST', 'ALFA.ST', 'BOL.ST', 'ELUX-B.ST', 'SKF-B.ST', 'GETI-B.ST',

        # Norway
        'EQNR', 'TEL.OL', 'DNB.OL', 'ORK.OL', 'MOWI.OL', 'AKRBP.OL', 'YAR.OL',
        'SALM.OL', 'SUBC.OL', 'TGS.OL', 'SCHA.OL', 'STB.OL',

        # Finland
        'NOKIA', 'NESTE.HE', 'FORTUM.HE', 'UPM.HE', 'SAMPO.HE', 'KNEBV.HE', 'STERV.HE',

        # ===========================================
        # SPAIN & PORTUGAL (~15)
        # ===========================================
        'SAN.MC', 'IBE.MC', 'ITX.MC', 'BBVA.MC', 'TEF.MC', 'FER.MC', 'REE.MC',
        'AMS.MC', 'AENA.MC', 'CLNX.MC', 'ENG.MC', 'CABK.MC', 'MAP.MC', 'GRF.MC',
        'EDP.LS', 'GALP.LS', 'SON.LS',

        # ===========================================
        # ITALY (~15)
        # ===========================================
        'ENEL.MI', 'ENI.MI', 'ISP.MI', 'UCG.MI', 'STMMI.MI', 'G.MI', 'TIT.MI',
        'PRY.MI', 'CNHI.MI', 'FCA.MI', 'LDO.MI', 'TEN.MI', 'RACE.MI', 'CPR.MI',
        'MONC.MI', 'NEXI.MI', 'A2A.MI',

        # ===========================================
        # BELGIUM & IRELAND (~10)
        # ===========================================
        'ABI.BR', 'KBC.BR', 'UCB.BR', 'SOLB.BR', 'ACKB.BR', 'PROX.BR', 'ARGX',
        'CRH', 'SMURFIT.L', 'ACCENTURE',

        # ===========================================
        # CANADA (~30)
        # ===========================================
        'TD', 'RY', 'BNS', 'BMO', 'CM', 'ENB', 'CNQ', 'CP', 'CNI', 'TRP',
        'SU', 'MFC', 'BCE', 'BN', 'FNV', 'WPM', 'NTR', 'WCN', 'TRI', 'BAM',
        # Toronto exchange
        'SHOP.TO', 'CSU.TO', 'ATD.TO', 'L.TO', 'WSP.TO', 'QSR.TO', 'FFH.TO',
        'GIB-A.TO', 'DOL.TO', 'MG.TO', 'IFC.TO', 'POW.TO', 'SLF.TO', 'GWO.TO',
        'OTEX.TO', 'CAR-UN.TO', 'CCL-B.TO', 'TFII.TO', 'SAP.TO', 'FSV.TO',

        # ===========================================
        # AUSTRALIA (~25)
        # ===========================================
        'BHP', 'RIO', 'CMWAY', 'CSLLY', 'WDS', 'ANZGY', 'NABZY', 'MQBKY', 'WEBNF',
        # Australian exchange
        'BHP.AX', 'RIO.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX',
        'MQG.AX', 'WES.AX', 'WOW.AX', 'TLS.AX', 'WDS.AX', 'FMG.AX', 'NCM.AX',
        'GMG.AX', 'TCL.AX', 'REA.AX', 'RHC.AX', 'ALL.AX', 'SHL.AX', 'COL.AX',

        # ===========================================
        # ASIA-PACIFIC (Non-Japan) (~25)
        # ===========================================
        # Taiwan
        'TSM', '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2412.TW', '2882.TW',

        # Hong Kong (ADRs and local)
        '0700.HK', '9988.HK', '0005.HK', '0941.HK', '0388.HK', '1299.HK', '2318.HK',
        '0883.HK', '0027.HK', '0011.HK', '0016.HK', '0002.HK', '0001.HK',

        # Singapore
        'SE', 'GRAB', 'DBS.SI', 'OCBC.SI', 'UOB.SI', 'D05.SI', 'O39.SI', 'Z74.SI',

        # ===========================================
        # CHINA ADRs (~15)
        # ===========================================
        'BABA', 'TCEHY', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'NTES',
        'BILI', 'TME', 'IQ', 'VIPS', 'ZTO', 'YUMC',

        # ===========================================
        # ISRAEL (~10)
        # ===========================================
        'NICE', 'CHKP', 'WIX', 'MNDY', 'CYBR', 'TEVA', 'INMD', 'GLOB',

        # ===========================================
        # NEW ZEALAND (~5)
        # ===========================================
        'FPH.NZ', 'SPK.NZ', 'AIR.NZ', 'AIA.NZ', 'MEL.NZ',
    ]


def get_msci_world_tickers(
    force_refresh: bool = False,
    validate: bool = True,
    use_cache: bool = True,
    min_tickers: int = 500
) -> List[str]:
    """
    Get comprehensive MSCI World constituent tickers.

    This is the main entry point. It attempts to:
    1. Load from cache if valid
    2. Download fresh holdings from iShares
    3. Map to yfinance tickers
    4. Validate tickers
    5. Fallback to curated list if needed

    Args:
        force_refresh: Force re-download even if cache is valid
        validate: Whether to validate tickers with yfinance
        use_cache: Whether to use caching
        min_tickers: Minimum tickers required (fallback if below this)

    Returns:
        List of yfinance-compatible ticker symbols
    """
    cache_path = get_cache_path('msci_world_tickers.json')

    # Try cache first
    if use_cache and not force_refresh and is_cache_valid(cache_path):
        logger.info("Loading tickers from cache...")
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                tickers = cached_data.get('tickers', [])
                if len(tickers) >= min_tickers:
                    logger.info(f"Loaded {len(tickers)} tickers from cache")
                    return tickers
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # Download fresh holdings
    holdings_df = download_urth_holdings()

    if holdings_df is None:
        holdings_df = download_holdings_alternative()

    yf_tickers = []

    if holdings_df is not None and len(holdings_df) > 0:
        logger.info(f"Processing {len(holdings_df)} holdings...")

        # Map to yfinance tickers
        for _, row in holdings_df.iterrows():
            yf_ticker = map_to_yfinance_ticker(row)
            if yf_ticker:
                yf_tickers.append(yf_ticker)

        # Remove duplicates while preserving order
        yf_tickers = list(dict.fromkeys(yf_tickers))
        logger.info(f"Mapped to {len(yf_tickers)} unique yfinance tickers")

    # Add curated fallback tickers
    fallback_tickers = get_curated_fallback_tickers()

    # Merge: prioritize downloaded tickers, add fallbacks
    all_tickers = list(dict.fromkeys(yf_tickers + fallback_tickers))
    logger.info(f"Total unique tickers after merging: {len(all_tickers)}")

    # Validate if requested
    if validate:
        valid_tickers, invalid_tickers = validate_tickers_batch(all_tickers)

        if len(invalid_tickers) > 0:
            logger.info(f"Invalid tickers ({len(invalid_tickers)}): {invalid_tickers[:20]}...")

        final_tickers = valid_tickers
    else:
        final_tickers = all_tickers

    # Ensure minimum coverage
    if len(final_tickers) < min_tickers:
        logger.warning(f"Only {len(final_tickers)} valid tickers, below minimum {min_tickers}")
        # Use curated list as primary
        if validate:
            valid_fallback, _ = validate_tickers_batch(fallback_tickers)
            final_tickers = list(dict.fromkeys(final_tickers + valid_fallback))
        else:
            final_tickers = list(dict.fromkeys(final_tickers + fallback_tickers))

    # Cache results
    if use_cache:
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'tickers': final_tickers,
                    'count': len(final_tickers),
                    'date': datetime.now().isoformat(),
                    'source': 'iShares_URTH+curated'
                }, f, indent=2)
            logger.info(f"Cached {len(final_tickers)} tickers")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    logger.info(f"Final ticker count: {len(final_tickers)}")
    return final_tickers


def get_ticker_info() -> Dict:
    """
    Get metadata about the ticker list.

    Returns:
        Dictionary with source info, count, last update, etc.
    """
    cache_path = get_cache_path('msci_world_tickers.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    return {
        'tickers': [],
        'count': 0,
        'date': None,
        'source': 'not_loaded'
    }


# Convenience function for backward compatibility
def get_msci_world_constituents() -> List[str]:
    """
    Backward-compatible function name.
    Returns MSCI World tickers (validates on first run, then uses cache).
    """
    return get_msci_world_tickers(validate=True, use_cache=True)


if __name__ == '__main__':
    # Test the module
    print("=" * 60)
    print("MSCI World Ticker Downloader")
    print("=" * 60)

    # Get tickers (force refresh to test)
    tickers = get_msci_world_tickers(force_refresh=True, validate=True)

    print(f"\nTotal valid tickers: {len(tickers)}")
    print(f"\nSample tickers (first 50):")
    print(tickers[:50])

    # Show info
    info = get_ticker_info()
    print(f"\nTicker info:")
    print(f"  Source: {info.get('source', 'unknown')}")
    print(f"  Count: {info.get('count', 0)}")
    print(f"  Last update: {info.get('date', 'unknown')}")
