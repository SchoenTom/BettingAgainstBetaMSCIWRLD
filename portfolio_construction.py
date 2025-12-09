"""
portfolio_construction.py - Construct BAB portfolios from prepared data

This script:
1. Loads saved CSV files (betas, returns)
2. Forms monthly cross-sections of stocks
3. Sorts stocks into beta quintiles (Q1 lowest, Q5 highest)
4. Constructs monthly BAB returns (Q1 - Q5 equal-weighted)
5. Produces summary statistics

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
NUM_QUINTILES = 5


def load_data():
    """
    Load required data from CSV files.

    Returns:
        tuple: (returns, excess_returns, betas) DataFrames
    """
    logger.info("Loading data files...")

    # Load monthly returns
    returns_path = os.path.join(DATA_DIR, 'monthly_returns.csv')
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded returns: {returns.shape}")

    # Load excess returns
    excess_returns_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')
    excess_returns = pd.read_csv(excess_returns_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded excess returns: {excess_returns.shape}")

    # Load rolling betas
    betas_path = os.path.join(DATA_DIR, 'rolling_betas.csv')
    betas = pd.read_csv(betas_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded betas: {betas.shape}")

    return returns, excess_returns, betas


def get_stock_columns(df):
    """
    Get list of stock columns (excluding benchmark).

    Args:
        df: DataFrame with tickers as columns

    Returns:
        list: Stock ticker columns
    """
    benchmark_cols = ['MSCI_World', 'RF_Rate']
    stock_cols = [col for col in df.columns if col not in benchmark_cols]
    return stock_cols


def assign_quintiles(betas_series, num_quintiles=NUM_QUINTILES):
    """
    Assign stocks to quintiles based on their betas.

    Q1 = Lowest beta (20% of stocks)
    Q5 = Highest beta (20% of stocks)

    Args:
        betas_series: Series of betas for a single month
        num_quintiles: Number of groups

    Returns:
        pd.Series: Quintile assignments (1 to 5)
    """
    # Drop NaN values
    valid_betas = betas_series.dropna()

    if len(valid_betas) < num_quintiles:
        return pd.Series(dtype=float)

    # Use qcut for equal-sized groups
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
    1. Use betas from month t-1 (lagged) to avoid look-ahead.
    2. Sort stocks into quintiles on beta_i,t-1.
    3. Compute equal-weight excess returns for Q1/Q5 in month t.
    4. Scale legs by 1/beta_L and 1/beta_H so ex-ante market beta ~ 0:
       r_BAB,t = (1/beta_L)*(r_L,t - Rf_t) - (1/beta_H)*(r_H,t - Rf_t)
       With excess returns, this is (1/beta_L)*r_L_excess - (1/beta_H)*r_H_excess.

    Ex-ante beta diagnostics are logged each month to verify neutrality.

    Args:
        excess_returns: DataFrame of monthly EXCESS returns
        betas: DataFrame of rolling betas

    Returns:
        pd.DataFrame: Monthly BAB portfolio data
    """
    logger.info("Constructing BAB portfolios...")
    logger.info("Universe caveat: using ~159 URTH-based tickers (proxy for MSCI World); selection and survivorship bias remain. Not full MSCI World (~1000 constituents).")

    stock_cols = get_stock_columns(excess_returns)
    logger.info(f"Stock columns in excess returns: {len(stock_cols)}")

    # Align DataFrames
    common_dates = excess_returns.index.intersection(betas.index)
    common_stocks = [col for col in stock_cols if col in betas.columns]

    logger.info(f"Common dates between returns and betas: {len(common_dates)}")
    logger.info(f"Common stocks: {len(common_stocks)}")

    if len(common_dates) == 0:
        logger.warning("No common dates found between returns and betas!")
        logger.info(f"Returns date range: {excess_returns.index.min()} to {excess_returns.index.max()}")
        logger.info(f"Betas date range: {betas.index.min()} to {betas.index.max()}")
        return pd.DataFrame(columns=[
            'BAB_Return', 'Q1_Mean_Beta', 'Q5_Mean_Beta',
            'Q1_Mean_Return', 'Q5_Mean_Return', 'N_Q1', 'N_Q5',
            'N_Total', 'Beta_Spread'
        ])

    if len(common_stocks) == 0:
        logger.warning("No common stocks found between returns and betas!")
        logger.info(f"Returns columns sample: {list(excess_returns.columns[:5])}")
        logger.info(f"Betas columns sample: {list(betas.columns[:5])}")
        return pd.DataFrame(columns=[
            'BAB_Return', 'Q1_Mean_Beta', 'Q5_Mean_Beta',
            'Q1_Mean_Return', 'Q5_Mean_Return', 'N_Q1', 'N_Q5',
            'N_Total', 'Beta_Spread'
        ])

    returns_aligned = excess_returns.loc[common_dates, common_stocks]
    betas_aligned = betas.loc[common_dates, common_stocks]

    # Results storage
    results = []
    portfolio_tickers = {}

    # Iterate through months (starting from second month for lagged beta)
    dates = common_dates.sort_values()
    logger.info(f"Processing {len(dates) - 1} months...")

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        # Get lagged betas (from previous month) - NO LOOK-AHEAD
        betas_lagged = betas_aligned.loc[prev_date]

        # Get current month returns
        returns_current = returns_aligned.loc[current_date]

        # Get stocks with valid beta AND return
        valid_stocks = betas_lagged.dropna().index.intersection(
            returns_current.dropna().index
        )

        if len(valid_stocks) < NUM_QUINTILES * 2:
            # Not enough stocks for meaningful quintiles
            continue

        # Assign quintiles based on lagged beta
        quintiles = assign_quintiles(betas_lagged[valid_stocks])

        if quintiles.empty:
            continue

        # Get Q1 (lowest beta) and Q5 (highest beta) stocks
        q1_stocks = quintiles[quintiles == 1].index
        q5_stocks = quintiles[quintiles == NUM_QUINTILES].index

        if len(q1_stocks) == 0 or len(q5_stocks) == 0:
            continue

        # Equal-weight excess returns (no market-cap data available)
        q1_weights = pd.Series(1 / len(q1_stocks), index=q1_stocks)
        q5_weights = pd.Series(1 / len(q5_stocks), index=q5_stocks)

        q1_return = (returns_current[q1_stocks] * q1_weights).sum()
        q5_return = (returns_current[q5_stocks] * q5_weights).sum()

        # Portfolio betas for the two legs (equal weights)
        q1_mean_beta = (betas_lagged[q1_stocks] * q1_weights).sum()
        q5_mean_beta = (betas_lagged[q5_stocks] * q5_weights).sum()

        beta_L = q1_mean_beta
        beta_H = q5_mean_beta

        if pd.isna(beta_L) or pd.isna(beta_H) or beta_L == 0 or beta_H == 0:
            continue

        # Beta-scaled BAB excess return (Frazzini-Pedersen)
        bab_return = (1 / beta_L) * q1_return - (1 / beta_H) * q5_return

        # Ex-ante beta diagnostic: scaled weights applied to betas
        long_effective = (1 / beta_L) * q1_weights
        short_effective = -(1 / beta_H) * q5_weights
        long_beta_ex = (betas_lagged[q1_stocks] * long_effective).sum()
        short_beta_ex = (betas_lagged[q5_stocks] * short_effective).sum()
        beta_ex_ante = long_beta_ex + short_beta_ex

        # Store results
        results.append({
            'Date': current_date,
            'BAB_Return': bab_return,  # excess return by construction
            'BAB_Excess_Return': bab_return,
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
            'Beta_Long_ExAnte': long_beta_ex,
            'Beta_Short_ExAnte': short_beta_ex
        })

        # Track tickers used at least once
        for ticker in q1_stocks.union(q5_stocks):
            portfolio_tickers[ticker] = portfolio_tickers.get(ticker, 0) + 1

        logger.info(f"{current_date.date()}: Q1 n={len(q1_stocks)}, beta={q1_mean_beta:.4f}; "
                    f"Q5 n={len(q5_stocks)}, beta={q5_mean_beta:.4f}; ex-ante beta={beta_ex_ante:.6f}")

    # Create DataFrame
    bab_df = pd.DataFrame(results)

    if bab_df.empty:
        logger.warning("No valid BAB portfolios could be constructed!")
        logger.warning(f"Common dates between returns and betas: {len(common_dates)}")
        logger.warning(f"Common stocks: {len(common_stocks)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'BAB_Return', 'Q1_Mean_Beta', 'Q5_Mean_Beta',
            'Q1_Mean_Return', 'Q5_Mean_Return', 'N_Q1', 'N_Q5',
            'N_Total', 'Beta_Spread'
        ])

    bab_df.set_index('Date', inplace=True)
    bab_df.index = pd.to_datetime(bab_df.index)

    # Ex-ante beta diagnostics
    beta_series = bab_df['Beta_BAB_ExAnte']
    mean_beta = beta_series.mean()
    abs_mean_beta = beta_series.abs().mean()
    logger.info(f"Constructed BAB portfolios: {len(bab_df)} months")
    logger.info(
        "Ex-ante beta diagnostics (target ~0): "
        f"mean={mean_beta:.6f}, abs-mean={abs_mean_beta:.6f}, "
        f"min={beta_series.min():.6f}, max={beta_series.max():.6f}, "
        f"std={beta_series.std():.6f}, p5={beta_series.quantile(0.05):.6f}, "
        f"p95={beta_series.quantile(0.95):.6f}"
    )
    if (beta_series.abs() > 0.10).any():
        logger.warning("Some rebalance months have |ex-ante beta| > 0.10. Check scaling and data quality.")
    if abs_mean_beta > 0.05:
        logger.warning("Average |ex-ante beta| exceeds 0.05; BAB may not be fully neutral.")

    # Save ticker usage for transparency
    ticker_df = pd.DataFrame([
        {'Ticker': t, 'Months_In_Portfolio': c} for t, c in sorted(portfolio_tickers.items())
    ])
    if not ticker_df.empty:
        ticker_df.to_csv(os.path.join(OUTPUT_DIR, 'portfolio_tickers.csv'), index=False)
        logger.info(f"Saved portfolio ticker usage to {os.path.join(OUTPUT_DIR, 'portfolio_tickers.csv')}")

    return bab_df


def compute_all_quintile_returns(excess_returns, betas):
    """
    Compute returns for all quintiles (Q1 to Q5).

    Useful for analyzing the full beta-return relationship.

    Args:
        excess_returns: DataFrame of monthly EXCESS returns
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
    dates = common_dates.sort_values()

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]

        betas_lagged = betas_aligned.loc[prev_date]
        returns_current = returns_aligned.loc[current_date]

        valid_stocks = betas_lagged.dropna().index.intersection(
            returns_current.dropna().index
        )

        if len(valid_stocks) < NUM_QUINTILES * 2:
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

    quintile_df = pd.DataFrame(results)

    if quintile_df.empty:
        logger.warning("No valid quintile returns could be computed!")
        # Return empty DataFrame with expected columns
        cols = ['Date']
        for q in range(1, NUM_QUINTILES + 1):
            cols.extend([f'Q{q}_Return', f'Q{q}_Mean_Beta', f'Q{q}_N'])
        return pd.DataFrame(columns=cols[1:])  # Exclude Date since it would be index

    quintile_df.set_index('Date', inplace=True)
    quintile_df.index = pd.to_datetime(quintile_df.index)

    return quintile_df


def save_outputs(bab_df, quintile_df):
    """
    Save portfolio construction outputs to CSV.

    Args:
        bab_df: BAB portfolio DataFrame
        quintile_df: All quintiles DataFrame
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    bab_df.to_csv(bab_path)
    logger.info(f"Saved BAB portfolio to {bab_path}")

    # Save all quintile returns
    quintile_path = os.path.join(OUTPUT_DIR, 'quintile_returns.csv')
    quintile_df.to_csv(quintile_path)
    logger.info(f"Saved quintile returns to {quintile_path}")


def print_summary(bab_df, quintile_df):
    """
    Print summary statistics.

    Args:
        bab_df: BAB portfolio DataFrame
        quintile_df: All quintiles DataFrame
    """
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

    print("\n--- Portfolio Size ---")
    print(f"Average Q1 Stocks: {bab_df['N_Q1'].mean():.1f}")
    print(f"Average Q5 Stocks: {bab_df['N_Q5'].mean():.1f}")
    print(f"Average Total:     {bab_df['N_Total'].mean():.1f}")

    print("\n--- Return Statistics (Monthly) ---")
    bab_mean = bab_df['BAB_Return'].mean()
    bab_std = bab_df['BAB_Return'].std()
    t_stat = bab_mean / (bab_std / np.sqrt(len(bab_df))) if bab_std > 0 else 0

    def stars(t):
        if abs(t) > 2.58:
            return "***"
        if abs(t) > 1.96:
            return "**"
        if abs(t) > 1.65:
            return "*"
        return ""

    print(f"BAB Mean Return:  {bab_mean*100:.3f}% {stars(t_stat)} (t={t_stat:.2f})")
    print(f"BAB Std Dev:      {bab_std*100:.3f}%")
    print(f"Q1 Mean Return:   {bab_df['Q1_Mean_Return'].mean()*100:.3f}%")
    print(f"Q5 Mean Return:   {bab_df['Q5_Mean_Return'].mean()*100:.3f}%")

    print("\n--- Annualized Statistics ---")
    ann_return = (1 + bab_df['BAB_Return'].mean()) ** 12 - 1
    ann_vol = bab_df['BAB_Return'].std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    print(f"BAB Annualized Return: {ann_return*100:.2f}%")
    print(f"BAB Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"BAB Sharpe Ratio:      {sharpe:.3f}")

    print("\n--- Ex-Ante Beta Neutrality Diagnostics ---")
    print(f"Mean Ex-Ante Beta:     {bab_df['Beta_BAB_ExAnte'].mean():.4f}")
    print(f"Abs-Mean Ex-Ante Beta: {bab_df['Beta_BAB_ExAnte'].abs().mean():.4f}")
    print(f"Min/Max Ex-Ante Beta:  {bab_df['Beta_BAB_ExAnte'].min():.4f} / {bab_df['Beta_BAB_ExAnte'].max():.4f}")

    print("\n" + "=" * 60)


def main():
    """
    Main portfolio construction pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting Portfolio Construction")
    logger.info("=" * 60)

    # Load data
    returns, excess_returns, betas = load_data()

    # Construct BAB portfolios using EXCESS returns
    bab_df = construct_bab_portfolios(excess_returns, betas)

    # Compute all quintile returns (excess)
    quintile_df = compute_all_quintile_returns(excess_returns, betas)

    # Save outputs
    save_outputs(bab_df, quintile_df)

    # Print summary
    print_summary(bab_df, quintile_df)

    logger.info("Portfolio construction complete!")


if __name__ == '__main__':
    main()
