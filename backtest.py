"""
backtest.py - Compute performance statistics for BAB strategy

This script:
1. Loads BAB portfolio returns and benchmark returns
2. Computes comprehensive performance statistics:
   - Annualized return, volatility, Sharpe ratio
   - Maximum drawdown, Sortino ratio, Calmar ratio
   - CAPM alpha and beta with t-statistics
   - Information ratio
3. Creates monthly performance DataFrame
4. Saves summary and detailed performance files

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import os
import logging
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import DATA_DIR, OUTPUT_DIR, PERIODS_PER_YEAR, ensure_directories


def load_data():
    """
    Load BAB portfolio and benchmark excess returns.

    Returns:
        tuple: (bab_df, benchmark_excess)
    """
    logger.info("Loading data...")

    # Load BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    if not os.path.exists(bab_path):
        raise FileNotFoundError(f"Missing {bab_path}. Run portfolio_construction.py first.")

    bab_df = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded BAB portfolio: {bab_df.shape}")

    if bab_df.empty:
        raise RuntimeError("BAB portfolio is empty! Check portfolio_construction.py output.")

    if 'BAB_Excess_Return' not in bab_df.columns:
        raise RuntimeError("BAB_Excess_Return column missing in bab_portfolio.csv")

    # Load benchmark excess returns
    excess_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')
    if not os.path.exists(excess_path):
        raise FileNotFoundError(f"Missing {excess_path}. Run data_loader.py first.")

    excess_returns = pd.read_csv(excess_path, index_col=0, parse_dates=True)

    if 'MSCI_World' in excess_returns.columns:
        benchmark = excess_returns['MSCI_World']
    else:
        logger.warning("MSCI_World not found, using first column as benchmark")
        benchmark = excess_returns.iloc[:, 0]

    logger.info(f"Loaded benchmark excess returns: {len(benchmark)} months")

    return bab_df, benchmark


def compute_cumulative_returns(returns):
    """Compute cumulative returns (growth of $1)."""
    return (1 + returns).cumprod()


def compute_annualized_return(returns, periods_per_year=PERIODS_PER_YEAR):
    """Compute annualized return from periodic returns."""
    return returns.mean() * periods_per_year


def compute_annualized_volatility(returns, periods_per_year=PERIODS_PER_YEAR):
    """Compute annualized volatility from periodic returns."""
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(returns, rf_rate=0, periods_per_year=PERIODS_PER_YEAR):
    """Compute Sharpe ratio (assumes returns are already excess if rf_rate=0)."""
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return 0
    return (ann_return - rf_rate) / ann_vol


def compute_max_drawdown(returns):
    """Compute maximum drawdown from a return series."""
    cum_returns = compute_cumulative_returns(returns)
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    return abs(drawdowns.min())


def compute_sortino_ratio(returns, rf_rate=0, periods_per_year=PERIODS_PER_YEAR):
    """Compute Sortino ratio (uses downside deviation)."""
    ann_return = compute_annualized_return(returns, periods_per_year)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0
    return (ann_return - rf_rate) / downside_std


def compute_calmar_ratio(returns, periods_per_year=PERIODS_PER_YEAR):
    """Compute Calmar ratio (return / max drawdown)."""
    ann_return = compute_annualized_return(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)
    if max_dd == 0:
        return 0
    return ann_return / max_dd


def compute_win_rate(returns):
    """Compute percentage of positive return periods."""
    return (returns > 0).mean()


def run_capm(excess_strategy, excess_benchmark):
    """
    Run CAPM regression: r_s = alpha + beta * r_m + e

    Returns monthly alpha, beta with t-stats, p-values, and R2.
    """
    common = excess_strategy.index.intersection(excess_benchmark.index)
    y = excess_strategy.loc[common].dropna()
    x = excess_benchmark.loc[common].dropna()
    common = y.index.intersection(x.index)
    y = y.loc[common]
    x = x.loc[common]

    n = len(y)
    if n < 12:
        logger.warning(f"Only {n} observations for CAPM regression (need at least 12)")
        return {
            'alpha_monthly': np.nan, 'beta_mkt': np.nan,
            'alpha_t': np.nan, 'beta_t': np.nan,
            'alpha_p': np.nan, 'beta_p': np.nan,
            'r2': np.nan, 'n': n
        }

    # OLS regression
    X = np.column_stack([np.ones(n), x.values])
    try:
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y.values)
    except np.linalg.LinAlgError:
        return {
            'alpha_monthly': np.nan, 'beta_mkt': np.nan,
            'alpha_t': np.nan, 'beta_t': np.nan,
            'alpha_p': np.nan, 'beta_p': np.nan,
            'r2': np.nan, 'n': n
        }

    residuals = y.values - X @ beta_hat
    sigma2 = (residuals @ residuals) / (n - 2)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta_hat / se_beta

    # P-values (two-tailed)
    def normal_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    p_vals = [2 * (1 - normal_cdf(abs(t))) for t in t_stats]

    # R-squared
    y_hat = X @ beta_hat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y.values - y_hat) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'alpha_monthly': beta_hat[0],
        'beta_mkt': beta_hat[1],
        'alpha_t': t_stats[0],
        'beta_t': t_stats[1],
        'alpha_p': p_vals[0],
        'beta_p': p_vals[1],
        'r2': r2,
        'n': n
    }


def compute_information_ratio(returns, benchmark_returns, periods_per_year=PERIODS_PER_YEAR):
    """Compute Information Ratio (excess return / tracking error)."""
    common_dates = returns.index.intersection(benchmark_returns.index)
    excess = returns.loc[common_dates] - benchmark_returns.loc[common_dates]
    ann_excess = compute_annualized_return(excess, periods_per_year)
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0
    return ann_excess / tracking_error


def compute_rolling_sharpe(returns, window=12, periods_per_year=PERIODS_PER_YEAR):
    """Compute rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window=window).mean() * periods_per_year
    rolling_std = returns.rolling(window=window).std() * np.sqrt(periods_per_year)
    return rolling_mean / rolling_std


def compute_all_statistics(bab_returns, benchmark_returns, strategy_name="BAB"):
    """
    Compute all performance statistics.

    Args:
        bab_returns: Series of BAB excess returns
        benchmark_returns: Series of benchmark excess returns
        strategy_name: Name for the strategy

    Returns:
        dict: Dictionary of performance statistics
    """
    logger.info(f"Computing statistics for {strategy_name}...")

    # Align dates
    common_dates = bab_returns.index.intersection(benchmark_returns.index)
    bab = bab_returns.loc[common_dates]
    bench = benchmark_returns.loc[common_dates]

    logger.info(f"Aligned data: {len(bab)} months")

    # T-statistic for mean return
    mean_ret = bab.mean()
    std_ret = bab.std()
    t_stat = mean_ret / (std_ret / np.sqrt(len(bab))) if std_ret > 0 else 0

    # CAPM regression
    capm = run_capm(bab, bench)

    stats = {
        # Basic info
        'Strategy': strategy_name,
        'Start_Date': bab.index.min().strftime('%Y-%m-%d'),
        'End_Date': bab.index.max().strftime('%Y-%m-%d'),
        'N_Months': len(bab),
        'Mean_Tstat': t_stat,

        # Return metrics
        'Total_Return': (1 + bab).prod() - 1,
        'Annualized_Return': compute_annualized_return(bab),
        'Annualized_Volatility': compute_annualized_volatility(bab),
        'Sharpe_Ratio': compute_sharpe_ratio(bab),
        'Sortino_Ratio': compute_sortino_ratio(bab),

        # Risk metrics
        'Max_Drawdown': compute_max_drawdown(bab),
        'Calmar_Ratio': compute_calmar_ratio(bab),
        'Skewness': bab.skew(),
        'Kurtosis': bab.kurtosis(),

        # Win rate
        'Win_Rate': compute_win_rate(bab),
        'Avg_Monthly_Return': bab.mean(),
        'Best_Month': bab.max(),
        'Worst_Month': bab.min(),

        # Relative to benchmark
        'Beta_to_Benchmark': capm['beta_mkt'],
        'Beta_t': capm['beta_t'],
        'Beta_p': capm['beta_p'],
        'Alpha_Monthly': capm['alpha_monthly'],
        'Alpha_Annualized': capm['alpha_monthly'] * 12 if not pd.isna(capm['alpha_monthly']) else np.nan,
        'Alpha_t': capm['alpha_t'],
        'Alpha_p': capm['alpha_p'],
        'Information_Ratio': compute_information_ratio(bab, bench),
        'CAPM_R2': capm['r2'],

        # Benchmark stats
        'Benchmark_Ann_Return': compute_annualized_return(bench),
        'Benchmark_Ann_Vol': compute_annualized_volatility(bench),
        'Benchmark_Sharpe': compute_sharpe_ratio(bench),
        'Benchmark_Max_DD': compute_max_drawdown(bench),
    }

    return stats


def create_monthly_performance(bab_df, benchmark_returns):
    """
    Create detailed monthly performance DataFrame.

    Args:
        bab_df: BAB portfolio DataFrame
        benchmark_returns: Benchmark return series

    Returns:
        pd.DataFrame: Monthly performance data
    """
    logger.info("Creating monthly performance table...")

    perf = bab_df.copy()

    # Ensure BAB_Return exists
    if 'BAB_Excess_Return' in perf.columns:
        perf['BAB_Return'] = perf['BAB_Excess_Return']

    # Add benchmark returns
    common_dates = perf.index.intersection(benchmark_returns.index)
    perf = perf.loc[common_dates]
    perf['Benchmark_Return'] = benchmark_returns.loc[common_dates]

    # Cumulative returns
    perf['BAB_Cumulative'] = compute_cumulative_returns(perf['BAB_Return'])
    perf['Benchmark_Cumulative'] = compute_cumulative_returns(perf['Benchmark_Return'])

    # Excess return over benchmark
    perf['Excess_Return'] = perf['BAB_Return'] - perf['Benchmark_Return']
    perf['Excess_Cumulative'] = compute_cumulative_returns(perf['Excess_Return'])

    # Rolling metrics
    perf['Rolling_12M_BAB_Sharpe'] = compute_rolling_sharpe(perf['BAB_Return'])
    perf['Rolling_12M_Benchmark_Sharpe'] = compute_rolling_sharpe(perf['Benchmark_Return'])

    # Drawdowns
    bab_cum = perf['BAB_Cumulative']
    perf['BAB_Drawdown'] = bab_cum / bab_cum.expanding().max() - 1

    bench_cum = perf['Benchmark_Cumulative']
    perf['Benchmark_Drawdown'] = bench_cum / bench_cum.expanding().max() - 1

    logger.info(f"Monthly performance: {len(perf)} months")

    return perf


def save_outputs(summary_stats, monthly_perf):
    """Save backtest outputs to CSV."""
    ensure_directories()

    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_path = os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Save monthly performance
    monthly_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    monthly_perf.to_csv(monthly_path)
    logger.info(f"Saved monthly performance to {monthly_path}")


def print_summary(stats):
    """Print formatted summary statistics."""
    print("\n" + "=" * 70)
    print("BAB Strategy Backtest Results")
    print("=" * 70)

    print(f"\nPeriod: {stats['Start_Date']} to {stats['End_Date']}")
    print(f"Total Months: {stats['N_Months']}")

    def stars(t):
        if pd.isna(t): return ""
        if abs(t) > 2.58: return "***"
        if abs(t) > 1.96: return "**"
        if abs(t) > 1.65: return "*"
        return ""

    print("\n" + "-" * 70)
    print("                           BAB Strategy    MSCI World")
    print("-" * 70)

    print(f"Annualized Return:         {stats['Annualized_Return']*100:>10.2f}%{stars(stats['Mean_Tstat']):>3}     {stats['Benchmark_Ann_Return']*100:>10.2f}%")
    print(f"Annualized Volatility:     {stats['Annualized_Volatility']*100:>10.2f}%     {stats['Benchmark_Ann_Vol']*100:>10.2f}%")
    print(f"Sharpe Ratio:              {stats['Sharpe_Ratio']:>10.3f}      {stats['Benchmark_Sharpe']:>10.3f}")
    print(f"Max Drawdown:              {stats['Max_Drawdown']*100:>10.2f}%     {stats['Benchmark_Max_DD']*100:>10.2f}%")

    print("\n" + "-" * 70)
    print("BAB Strategy Details")
    print("-" * 70)

    print(f"Total Return:              {stats['Total_Return']*100:.2f}%")
    print(f"Sortino Ratio:             {stats['Sortino_Ratio']:.3f}")
    print(f"Calmar Ratio:              {stats['Calmar_Ratio']:.3f}")
    print(f"Win Rate:                  {stats['Win_Rate']*100:.1f}%")
    print(f"Best Month:                {stats['Best_Month']*100:.2f}%")
    print(f"Worst Month:               {stats['Worst_Month']*100:.2f}%")
    print(f"Skewness:                  {stats['Skewness']:.3f}")
    print(f"Kurtosis:                  {stats['Kurtosis']:.3f}")

    print("\n" + "-" * 70)
    print("Risk-Adjusted Metrics (CAPM)")
    print("-" * 70)

    beta = stats['Beta_to_Benchmark']
    alpha_ann = stats['Alpha_Annualized']

    if not pd.isna(beta):
        print(f"Beta to Benchmark:         {beta:.3f} (t={stats['Beta_t']:.2f}, p={stats['Beta_p']:.4f})")
    else:
        print(f"Beta to Benchmark:         N/A")

    if not pd.isna(alpha_ann):
        print(f"Alpha (Monthly):           {stats['Alpha_Monthly']*100:.2f}% (t={stats['Alpha_t']:.2f}, p={stats['Alpha_p']:.4f})")
        print(f"Alpha (Annualized):        {alpha_ann*100:.2f}%")
    else:
        print(f"Alpha:                     N/A")

    print(f"Information Ratio:         {stats['Information_Ratio']:.3f}")
    if not pd.isna(stats['CAPM_R2']):
        print(f"CAPM R^2:                  {stats['CAPM_R2']:.3f}")

    print("\n" + "=" * 70)


def main():
    """Main backtest pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Backtest Analysis")
    logger.info("=" * 60)

    ensure_directories()

    # Load data
    bab_df, benchmark = load_data()

    # Compute all statistics
    stats = compute_all_statistics(bab_df['BAB_Excess_Return'], benchmark)

    # Create monthly performance
    monthly_perf = create_monthly_performance(bab_df, benchmark)

    # Save outputs
    save_outputs(stats, monthly_perf)

    # Print summary
    print_summary(stats)

    logger.info("Backtest analysis complete!")


if __name__ == '__main__':
    main()
