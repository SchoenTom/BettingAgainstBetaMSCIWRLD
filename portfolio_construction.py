"""
portfolio_construction.py - Construct BAB portfolios from prepared data

This script implements the Betting Against Beta (BAB) strategy:
1. Loads betas and excess returns from data_loader.py output
2. Each month, sorts stocks into quintiles by lagged beta (t-1)
3. Forms equal-weighted portfolios for each quintile
4. Constructs BAB as: (1/beta_L)*Q1_excess - (1/beta_H)*Q5_excess
5. This beta-scaling makes the strategy approximately market-neutral

Key Design:
- Uses LAGGED betas (month t-1) to avoid look-ahead bias
- Equal-weighted within quintiles (no market cap data)
- Beta-scaled long/short positions for neutrality

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import (
    DATA_DIR, OUTPUT_DIR, NUM_QUINTILES, MIN_STOCKS_PER_QUINTILE,
    ensure_directories
)


def load_data():
    """
    Load required data from CSV files.

    Returns:
        tuple: (excess_returns, betas) DataFrames
    """
    logger.info("Loading data files...")

    # Load excess returns
    excess_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')
    if not os.path.exists(excess_path):
        raise FileNotFoundError(f"Missing {excess_path}. Run data_loader.py first.")
    excess_returns = pd.read_csv(excess_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded excess returns: {excess_returns.shape}")

    # Load rolling betas
    betas_path = os.path.join(DATA_DIR, 'rolling_betas.csv')
    if not os.path.exists(betas_path):
        raise FileNotFoundError(f"Missing {betas_path}. Run data_loader.py first.")
    betas = pd.read_csv(betas_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded betas: {betas.shape}")

    return excess_returns, betas


def get_stock_columns(df):
    """Get list of stock columns (excluding benchmark)."""
    exclude = ['MSCI_World', 'RF_Rate']
    return [col for col in df.columns if col not in exclude]


def assign_quintiles(betas_series, num_quintiles=NUM_QUINTILES):
    """
    Assign stocks to quintiles based on their betas.

    Q1 = Lowest beta (bottom 20%)
    Q5 = Highest beta (top 20%)

    Args:
        betas_series: Series of betas for a single month
        num_quintiles: Number of groups

    Returns:
        pd.Series: Quintile assignments (1 to 5)
    """
    valid_betas = betas_series.dropna()

    if len(valid_betas) < num_quintiles * MIN_STOCKS_PER_QUINTILE:
        return pd.Series(dtype=float)

    try:
        quintiles = pd.qcut(valid_betas, q=num_quintiles, labels=False, duplicates='drop') + 1
        return quintiles
    except ValueError:
        # Handle case with too few unique values
        return pd.Series(dtype=float)


def construct_bab_portfolios(excess_returns, betas):
    """
    Construct monthly BAB portfolios using excess returns and beta scaling.

    For each month t:
    1. Use betas from month t-1 (lagged) to avoid look-ahead
    2. Sort stocks into quintiles on beta_i,t-1
    3. Compute equal-weight excess returns for Q1 (low beta) and Q5 (high beta)
    4. Scale legs by 1/beta_L and 1/beta_H so ex-ante market beta ~ 0:
       BAB_excess = (1/beta_L) * Q1_excess - (1/beta_H) * Q5_excess

    Args:
        excess_returns: DataFrame of monthly excess returns
        betas: DataFrame of rolling betas

    Returns:
        pd.DataFrame: Monthly BAB portfolio data
    """
    logger.info("Constructing BAB portfolios...")

    stock_cols = get_stock_columns(excess_returns)
    logger.info(f"Stock columns in excess returns: {len(stock_cols)}")

    # Find common dates and stocks
    common_dates = excess_returns.index.intersection(betas.index)
    common_stocks = [col for col in stock_cols if col in betas.columns]

    logger.info(f"Common dates: {len(common_dates)}")
    logger.info(f"Common stocks: {len(common_stocks)}")

    if len(common_dates) < 2:
        logger.error("Not enough common dates between returns and betas!")
        logger.error(f"Returns date range: {excess_returns.index.min()} to {excess_returns.index.max()}")
        logger.error(f"Betas date range: {betas.index.min()} to {betas.index.max()}")
        return pd.DataFrame()

    if len(common_stocks) == 0:
        logger.error("No common stocks between returns and betas!")
        return pd.DataFrame()

    returns_aligned = excess_returns.loc[common_dates, common_stocks]
    betas_aligned = betas.loc[common_dates, common_stocks]

    # Results storage
    results = []
    portfolio_tickers = {}

    # Sort dates
    dates = sorted(common_dates)
    logger.info(f"Processing {len(dates) - 1} months (using lagged betas)...")

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        # Get LAGGED betas (from previous month) - NO LOOK-AHEAD
        betas_lagged = betas_aligned.loc[prev_date]

        # Get current month excess returns
        returns_current = returns_aligned.loc[current_date]

        # Get stocks with valid beta AND return
        valid_mask = betas_lagged.notna() & returns_current.notna()
        valid_stocks = valid_mask[valid_mask].index

        if len(valid_stocks) < NUM_QUINTILES * MIN_STOCKS_PER_QUINTILE:
            continue

        # Assign quintiles based on lagged beta
        quintiles = assign_quintiles(betas_lagged[valid_stocks])

        if quintiles.empty:
            continue

        # Get Q1 (lowest beta) and Q5 (highest beta) stocks
        q1_stocks = quintiles[quintiles == 1].index
        q5_stocks = quintiles[quintiles == NUM_QUINTILES].index

        if len(q1_stocks) < MIN_STOCKS_PER_QUINTILE or len(q5_stocks) < MIN_STOCKS_PER_QUINTILE:
            continue

        # Equal-weight excess returns
        q1_weights = pd.Series(1.0 / len(q1_stocks), index=q1_stocks)
        q5_weights = pd.Series(1.0 / len(q5_stocks), index=q5_stocks)

        q1_return = (returns_current[q1_stocks] * q1_weights).sum()
        q5_return = (returns_current[q5_stocks] * q5_weights).sum()

        # Portfolio betas
        q1_mean_beta = (betas_lagged[q1_stocks] * q1_weights).sum()
        q5_mean_beta = (betas_lagged[q5_stocks] * q5_weights).sum()

        beta_L = q1_mean_beta
        beta_H = q5_mean_beta

        # Skip if betas are invalid
        if pd.isna(beta_L) or pd.isna(beta_H) or beta_L <= 0 or beta_H <= 0:
            continue

        # Beta-Neutral BAB with Normalized Leverage
        #
        # Original F&P formula: (1/β_L)*r_L - (1/β_H)*r_H
        # Creates ex-ante beta = 0, but has implicit long bias (~$0.96 net long)
        #
        # Fix: Normalize total gross exposure to $2 (=$1 long + $1 short equivalent)
        # This maintains beta-neutrality while controlling leverage
        #
        # Raw weights: w_L_raw = 1/β_L, w_H_raw = 1/β_H
        # Gross exposure: w_L_raw + w_H_raw
        # Normalized weights: w_L = 2 * w_L_raw / (w_L_raw + w_H_raw)
        #                     w_H = 2 * w_H_raw / (w_L_raw + w_H_raw)
        # This ensures: w_L + w_H = 2 (total gross = $2)
        # And: w_L/β_L = w_H/β_H (beta neutral)

        w_L_raw = 1.0 / beta_L
        w_H_raw = 1.0 / beta_H
        gross_raw = w_L_raw + w_H_raw

        # Normalize to $2 gross exposure ($1 equivalent per leg)
        w_L = 2.0 * w_L_raw / gross_raw  # Normalized long weight
        w_H = 2.0 * w_H_raw / gross_raw  # Normalized short weight

        # Beta-neutral BAB return with controlled leverage
        bab_excess_return = w_L * q1_return - w_H * q5_return

        # Ex-ante beta should be ~0 (beta-neutral)
        beta_ex_ante = w_L * q1_mean_beta - w_H * q5_mean_beta

        # Track dollar weights for diagnostics
        dollar_long = w_L
        dollar_short = w_H

        # Store results
        results.append({
            'Date': current_date,
            'BAB_Return': bab_excess_return,
            'BAB_Excess_Return': bab_excess_return,
            'Q1_Mean_Beta': q1_mean_beta,
            'Q5_Mean_Beta': q5_mean_beta,
            'Q1_Mean_Return': q1_return,
            'Q5_Mean_Return': q5_return,
            'N_Q1': len(q1_stocks),
            'N_Q5': len(q5_stocks),
            'N_Total': len(valid_stocks),
            'Beta_Spread': q5_mean_beta - q1_mean_beta,
            'Beta_L': beta_L,
            'Beta_H': beta_H,
            'Beta_BAB_ExAnte': beta_ex_ante,
            'Dollar_Long': dollar_long,
            'Dollar_Short': dollar_short,
            'Dollar_Net': dollar_long - dollar_short,  # Should be 0 for dollar-neutral
        })

        # Track tickers used
        for ticker in list(q1_stocks) + list(q5_stocks):
            portfolio_tickers[ticker] = portfolio_tickers.get(ticker, 0) + 1

    # Create DataFrame
    if not results:
        logger.warning("No valid BAB portfolios could be constructed!")
        return pd.DataFrame()

    bab_df = pd.DataFrame(results)
    bab_df.set_index('Date', inplace=True)
    bab_df.index = pd.to_datetime(bab_df.index)

    logger.info(f"Constructed BAB portfolios: {len(bab_df)} months")
    logger.info(f"Date range: {bab_df.index.min().strftime('%Y-%m-%d')} to {bab_df.index.max().strftime('%Y-%m-%d')}")

    # Ex-ante beta diagnostics
    beta_series = bab_df['Beta_BAB_ExAnte']
    logger.info(f"Ex-ante beta stats: mean={beta_series.mean():.6f}, std={beta_series.std():.6f}")

    # Save ticker usage
    if portfolio_tickers:
        ticker_df = pd.DataFrame([
            {'Ticker': t, 'Months_In_Portfolio': c}
            for t, c in sorted(portfolio_tickers.items(), key=lambda x: -x[1])
        ])
        ensure_directories()
        ticker_df.to_csv(os.path.join(OUTPUT_DIR, 'portfolio_tickers.csv'), index=False)
        logger.info(f"Saved portfolio tickers ({len(ticker_df)} unique stocks used)")

    return bab_df


def compute_all_quintile_returns(excess_returns, betas):
    """
    Compute returns for all quintiles (Q1 to Q5).

    Args:
        excess_returns: DataFrame of monthly excess returns
        betas: DataFrame of rolling betas

    Returns:
        pd.DataFrame: Monthly returns for each quintile
    """
    logger.info("Computing all quintile returns...")

    stock_cols = get_stock_columns(excess_returns)
    common_dates = excess_returns.index.intersection(betas.index)
    common_stocks = [col for col in stock_cols if col in betas.columns]

    returns_aligned = excess_returns.loc[common_dates, common_stocks]
    betas_aligned = betas.loc[common_dates, common_stocks]

    results = []
    dates = sorted(common_dates)

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        betas_lagged = betas_aligned.loc[prev_date]
        returns_current = returns_aligned.loc[current_date]

        valid_mask = betas_lagged.notna() & returns_current.notna()
        valid_stocks = valid_mask[valid_mask].index

        if len(valid_stocks) < NUM_QUINTILES * MIN_STOCKS_PER_QUINTILE:
            continue

        quintiles = assign_quintiles(betas_lagged[valid_stocks])

        if quintiles.empty:
            continue

        row = {'Date': current_date}

        for q in range(1, NUM_QUINTILES + 1):
            q_stocks = quintiles[quintiles == q].index
            if len(q_stocks) > 0:
                row[f'Q{q}_Return'] = returns_current[q_stocks].mean()
                row[f'Q{q}_Mean_Beta'] = betas_lagged[q_stocks].mean()
                row[f'Q{q}_N'] = len(q_stocks)
            else:
                row[f'Q{q}_Return'] = np.nan
                row[f'Q{q}_Mean_Beta'] = np.nan
                row[f'Q{q}_N'] = 0

        results.append(row)

    if not results:
        logger.warning("No valid quintile returns computed!")
        return pd.DataFrame()

    quintile_df = pd.DataFrame(results)
    quintile_df.set_index('Date', inplace=True)
    quintile_df.index = pd.to_datetime(quintile_df.index)

    logger.info(f"Computed quintile returns: {len(quintile_df)} months")

    return quintile_df


def save_outputs(bab_df, quintile_df):
    """Save portfolio construction outputs to CSV."""
    ensure_directories()

    if not bab_df.empty:
        bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
        bab_df.to_csv(bab_path)
        logger.info(f"Saved BAB portfolio to {bab_path}")

    if not quintile_df.empty:
        quintile_path = os.path.join(OUTPUT_DIR, 'quintile_returns.csv')
        quintile_df.to_csv(quintile_path)
        logger.info(f"Saved quintile returns to {quintile_path}")


def print_summary(bab_df, quintile_df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("BAB Portfolio Construction Summary")
    print("=" * 60)

    if bab_df.empty:
        print("\nNo BAB portfolios were constructed. Check data alignment.")
        print("=" * 60)
        return

    print(f"\nDate Range: {bab_df.index.min().strftime('%Y-%m-%d')} to {bab_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Total Months: {len(bab_df)}")

    print("\n--- Beta Statistics ---")
    print(f"Average Q1 (Low) Beta:  {bab_df['Q1_Mean_Beta'].mean():.3f}")
    print(f"Average Q5 (High) Beta: {bab_df['Q5_Mean_Beta'].mean():.3f}")
    print(f"Average Beta Spread:    {bab_df['Beta_Spread'].mean():.3f}")

    print("\n--- Leverage & Exposure ---")
    if 'Dollar_Long' in bab_df.columns:
        print(f"Avg Long Weight:        ${bab_df['Dollar_Long'].mean():.2f}")
        print(f"Avg Short Weight:       ${bab_df['Dollar_Short'].mean():.2f}")
        print(f"Gross Exposure:         ${bab_df['Dollar_Long'].mean() + bab_df['Dollar_Short'].mean():.2f} (normalized to $2)")
        print(f"Net Dollar Exposure:    ${bab_df['Dollar_Net'].mean():.3f}")

    print("\n--- Ex-Ante Beta ---")
    if 'Beta_BAB_ExAnte' in bab_df.columns:
        beta_mean = bab_df['Beta_BAB_ExAnte'].mean()
        print(f"Ex-Ante Beta:           {beta_mean:.4f} (should be ~0 for beta-neutral)")

    print("\n--- Portfolio Size ---")
    print(f"Average Q1 Stocks: {bab_df['N_Q1'].mean():.1f}")
    print(f"Average Q5 Stocks: {bab_df['N_Q5'].mean():.1f}")
    print(f"Average Total:     {bab_df['N_Total'].mean():.1f}")

    print("\n--- Return Statistics (Monthly) ---")
    bab_mean = bab_df['BAB_Excess_Return'].mean()
    bab_std = bab_df['BAB_Excess_Return'].std()
    t_stat = bab_mean / (bab_std / np.sqrt(len(bab_df))) if bab_std > 0 else 0

    def stars(t):
        if abs(t) > 2.58: return "***"
        if abs(t) > 1.96: return "**"
        if abs(t) > 1.65: return "*"
        return ""

    print(f"BAB Mean Excess Return: {bab_mean*100:.3f}% {stars(t_stat)} (t={t_stat:.2f})")
    print(f"BAB Std Dev:            {bab_std*100:.3f}%")
    print(f"Q1 Mean Excess Return:  {bab_df['Q1_Mean_Return'].mean()*100:.3f}%")
    print(f"Q5 Mean Excess Return:  {bab_df['Q5_Mean_Return'].mean()*100:.3f}%")

    print("\n--- Annualized Statistics ---")
    ann_return = bab_mean * 12
    ann_vol = bab_std * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    print(f"BAB Annualized Return: {ann_return*100:.2f}%")
    print(f"BAB Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"BAB Sharpe Ratio:      {sharpe:.3f}")

    print("\n" + "=" * 60)


def main():
    """Main portfolio construction pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Portfolio Construction")
    logger.info("=" * 60)

    ensure_directories()

    # Load data
    excess_returns, betas = load_data()

    # Construct BAB portfolios
    bab_df = construct_bab_portfolios(excess_returns, betas)

    # Compute all quintile returns
    quintile_df = compute_all_quintile_returns(excess_returns, betas)

    # Save outputs
    save_outputs(bab_df, quintile_df)

    # Print summary
    print_summary(bab_df, quintile_df)

    logger.info("Portfolio construction complete!")


if __name__ == '__main__':
    main()
