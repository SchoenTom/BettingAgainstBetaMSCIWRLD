"""
backtest.py - Compute performance statistics for BAB strategy

This script:
1. Loads BAB portfolio returns and benchmark returns
2. Computes performance statistics:
   - Annualized return
   - Annualized volatility
   - Sharpe ratio
   - Max drawdown
   - Various other metrics
3. Saves summary and detailed performance files

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

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def load_data():
    """
    Load BAB portfolio and benchmark EXCESS returns.

    Returns:
        tuple: (bab_df, benchmark_excess)
    """
    logger.info("Loading data...")

    # Load BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    bab_df = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded BAB portfolio: {bab_df.shape}")

    if 'BAB_Excess_Return' not in bab_df.columns:
        raise RuntimeError("BAB_Excess_Return missing in bab_portfolio.csv; BAB must be based on excess returns.")

    # Load benchmark EXCESS returns
    excess_path = os.path.join(DATA_DIR, 'monthly_excess_returns.csv')
    excess_returns = pd.read_csv(excess_path, index_col=0, parse_dates=True)

    if 'MSCI_World' in excess_returns.columns:
        benchmark = excess_returns['MSCI_World']
    else:
        benchmark = excess_returns.iloc[:, 0]
        logger.warning("MSCI_World not found in monthly_excess_returns, using first column as benchmark")

    logger.info(f"Loaded benchmark EXCESS returns: {len(benchmark)}")

    return bab_df, benchmark


def compute_cumulative_returns(returns):
    """
    Compute cumulative returns from a return series.

    Args:
        returns: Series of periodic returns

    Returns:
        pd.Series: Cumulative returns (growth of $1)
    """
    return (1 + returns).cumprod()


def compute_annualized_return(returns, periods_per_year=12):
    """
    Compute annualized return from periodic returns.
    For excess returns, this is equivalent to mean * periods_per_year.

    Args:
        returns: Series of periodic returns (typically excess)
        periods_per_year: Number of periods per year

    Returns:
        float: Annualized return
    """
    return returns.mean() * periods_per_year


def compute_annualized_volatility(returns, periods_per_year=12):
    """
    Compute annualized volatility from periodic returns.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year

    Returns:
        float: Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(returns, rf_rate=0, periods_per_year=12):
    """
    Compute Sharpe ratio.

    Args:
        returns: Series of periodic returns
        rf_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year

    Returns:
        float: Sharpe ratio
    """
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return 0
    return (ann_return - rf_rate) / ann_vol


def compute_max_drawdown(returns):
    """
    Compute maximum drawdown from a return series.

    Args:
        returns: Series of periodic returns

    Returns:
        float: Maximum drawdown (as positive number)
    """
    cum_returns = compute_cumulative_returns(returns)
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    return abs(drawdowns.min())


def compute_sortino_ratio(returns, rf_rate=0, periods_per_year=12):
    """
    Compute Sortino ratio (uses downside deviation).

    Args:
        returns: Series of periodic returns
        rf_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        float: Sortino ratio
    """
    ann_return = compute_annualized_return(returns, periods_per_year)

    # Downside returns only
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return 0

    return (ann_return - rf_rate) / downside_std


def compute_calmar_ratio(returns, periods_per_year=12):
    """
    Compute Calmar ratio (return / max drawdown).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year

    Returns:
        float: Calmar ratio
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    max_dd = compute_max_drawdown(returns)

    if max_dd == 0:
        return 0

    return ann_return / max_dd


def compute_win_rate(returns):
    """
    Compute percentage of positive return periods.

    Args:
        returns: Series of periodic returns

    Returns:
        float: Win rate (0 to 1)
    """
    return (returns > 0).mean()


def compute_skewness(returns):
    """
    Compute skewness of returns.

    Args:
        returns: Series of periodic returns

    Returns:
        float: Skewness
    """
    return returns.dropna().skew()


def compute_kurtosis(returns):
    """
    Compute excess kurtosis of returns.

    Args:
        returns: Series of periodic returns

    Returns:
        float: Excess kurtosis
    """
    return returns.dropna().kurtosis()


def compute_rolling_sharpe(returns, window=12, periods_per_year=12):
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Series of periodic returns
        window: Rolling window size
        periods_per_year: Number of periods per year

    Returns:
        pd.Series: Rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window=window).mean() * periods_per_year
    rolling_std = returns.rolling(window=window).std() * np.sqrt(periods_per_year)

    return rolling_mean / rolling_std


def compute_information_ratio(returns, benchmark_returns, periods_per_year=12):
    """
    Compute Information Ratio (excess return / tracking error).

    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        float: Information ratio
    """
    # Align dates
    common_dates = returns.index.intersection(benchmark_returns.index)
    excess = returns.loc[common_dates] - benchmark_returns.loc[common_dates]

    ann_excess = compute_annualized_return(excess, periods_per_year)
    tracking_error = excess.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0

    return ann_excess / tracking_error


def run_capm(excess_strategy, excess_benchmark):
    """
    CAPM regression on EXCESS returns: r_s = alpha + beta * r_m + e
    Returns monthly alpha/beta with t-stats/p-values and R2.
    """
    common = excess_strategy.index.intersection(excess_benchmark.index)
    y = excess_strategy.loc[common].dropna()
    x = excess_benchmark.loc[common].dropna()
    common = y.index.intersection(x.index)
    y = y.loc[common]
    x = x.loc[common]

    n = len(y)
    if n < 24:
        raise RuntimeError("Not enough observations for CAPM regression.")

    X = np.column_stack([np.ones(n), x.values])
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y.values)
    residuals = y.values - X @ beta_hat
    sigma2 = (residuals @ residuals) / (n - 2)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta_hat / se_beta

    normal_cdf = np.vectorize(lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2))))
    p_vals = 2 * (1 - normal_cdf(np.abs(t_stats)))

    y_hat = X @ beta_hat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

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


def compute_beta_to_benchmark(returns, benchmark_returns):
    """
    Compute beta of strategy to benchmark.

    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns

    Returns:
        float: Beta
    """
    common_dates = returns.index.intersection(benchmark_returns.index)
    strat = returns.loc[common_dates].dropna()
    bench = benchmark_returns.loc[common_dates].dropna()

    common = strat.index.intersection(bench.index)
    strat = strat.loc[common]
    bench = bench.loc[common]

    covar = np.cov(strat, bench)[0, 1]
    var = np.var(bench)

    if var == 0:
        return 0

    return covar / var


def compute_alpha(returns, benchmark_returns, periods_per_year=12):
    """
    Compute alpha (Jensen's alpha).

    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        float: Annualized alpha
    """
    common_dates = returns.index.intersection(benchmark_returns.index)
    strat_ann = compute_annualized_return(returns.loc[common_dates], periods_per_year)
    bench_ann = compute_annualized_return(benchmark_returns.loc[common_dates], periods_per_year)
    beta = compute_beta_to_benchmark(returns, benchmark_returns)

    # Alpha = strategy return - beta * benchmark return
    return strat_ann - beta * bench_ann


def compute_all_statistics(bab_returns, benchmark_returns, strategy_name="BAB"):
    """
    Compute all performance statistics.

    Args:
        bab_returns: Series of BAB returns
        benchmark_returns: Series of benchmark returns
        strategy_name: Name for the strategy

    Returns:
        dict: Dictionary of performance statistics
    """
    logger.info(f"Computing statistics for {strategy_name}...")

    # Align dates
    common_dates = bab_returns.index.intersection(benchmark_returns.index)
    bab = bab_returns.loc[common_dates]
    bench = benchmark_returns.loc[common_dates]

    # Simple t-stat for mean monthly excess return (significance marker)
    mean_ret = bab.mean()
    std_ret = bab.std()
    t_stat = mean_ret / (std_ret / np.sqrt(len(bab))) if std_ret > 0 else 0

    capm = run_capm(bab, bench)

    stats_dict = {
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
        'Skewness': compute_skewness(bab),
        'Kurtosis': compute_kurtosis(bab),

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
        'Alpha_Annualized': capm['alpha_monthly'] * 12,
        'Alpha_t': capm['alpha_t'],
        'Alpha_p': capm['alpha_p'],
        'Information_Ratio': compute_information_ratio(bab, bench),
        'CAPM_R2': capm['r2'],

        # Benchmark stats for comparison
        'Benchmark_Ann_Return': compute_annualized_return(bench),
        'Benchmark_Ann_Vol': compute_annualized_volatility(bench),
        'Benchmark_Sharpe': compute_sharpe_ratio(bench),
        'Benchmark_Max_DD': compute_max_drawdown(bench),
    }

    return stats_dict


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

    # Start with BAB data
    perf = bab_df.copy()
    if 'BAB_Excess_Return' in perf.columns:
        perf['BAB_Return'] = perf['BAB_Excess_Return']

    # Add benchmark returns
    common_dates = perf.index.intersection(benchmark_returns.index)
    perf = perf.loc[common_dates]
    perf['Benchmark_Return'] = benchmark_returns.loc[common_dates]

    # Compute cumulative returns
    perf['BAB_Cumulative'] = compute_cumulative_returns(perf['BAB_Return'])
    perf['Benchmark_Cumulative'] = compute_cumulative_returns(perf['Benchmark_Return'])

    # Compute excess return over benchmark
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

    return perf


def save_outputs(summary_stats, monthly_perf):
    """
    Save backtest outputs to CSV.

    Args:
        summary_stats: Dictionary of summary statistics
        monthly_perf: DataFrame of monthly performance
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_path = os.path.join(OUTPUT_DIR, 'bab_backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Save monthly performance
    monthly_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    monthly_perf.to_csv(monthly_path)
    logger.info(f"Saved monthly performance to {monthly_path}")


def print_summary(stats_dict):
    """
    Print formatted summary statistics.

    Args:
        stats_dict: Dictionary of performance statistics
    """
    print("\n" + "=" * 70)
    print("BAB Strategy Backtest Results")
    print("=" * 70)

    print(f"\nPeriod: {stats_dict['Start_Date']} to {stats_dict['End_Date']}")
    print(f"Total Months: {stats_dict['N_Months']}")

    def stars(t):
        if abs(t) > 2.58:
            return "***"
        if abs(t) > 1.96:
            return "**"
        if abs(t) > 1.65:
            return "*"
        return ""

    print("\n" + "-" * 70)
    print("                           BAB Strategy    MSCI World")
    print("-" * 70)

    print(f"Annualized Return:         {stats_dict['Annualized_Return']*100:>10.2f}%{stars(stats_dict['Mean_Tstat']):>3}     {stats_dict['Benchmark_Ann_Return']*100:>10.2f}%")
    print(f"Annualized Volatility:     {stats_dict['Annualized_Volatility']*100:>10.2f}%     {stats_dict['Benchmark_Ann_Vol']*100:>10.2f}%")
    print(f"Sharpe Ratio:              {stats_dict['Sharpe_Ratio']:>10.3f}      {stats_dict['Benchmark_Sharpe']:>10.3f}")
    print(f"Max Drawdown:              {stats_dict['Max_Drawdown']*100:>10.2f}%     {stats_dict['Benchmark_Max_DD']*100:>10.2f}%")

    print("\n" + "-" * 70)
    print("BAB Strategy Details")
    print("-" * 70)

    print(f"Total Return:              {stats_dict['Total_Return']*100:.2f}%")
    print(f"Sortino Ratio:             {stats_dict['Sortino_Ratio']:.3f}")
    print(f"Calmar Ratio:              {stats_dict['Calmar_Ratio']:.3f}")
    print(f"Win Rate:                  {stats_dict['Win_Rate']*100:.1f}%")
    print(f"Best Month:                {stats_dict['Best_Month']*100:.2f}%")
    print(f"Worst Month:               {stats_dict['Worst_Month']*100:.2f}%")
    print(f"Skewness:                  {stats_dict['Skewness']:.3f}")
    print(f"Kurtosis:                  {stats_dict['Kurtosis']:.3f}")

    print("\n" + "-" * 70)
    print("Risk-Adjusted Metrics")
    print("-" * 70)

    print(f"Beta to Benchmark:         {stats_dict['Beta_to_Benchmark']:.3f} (t={stats_dict['Beta_t']:.2f}, p={stats_dict['Beta_p']:.4f})")
    print(f"Alpha (Monthly):           {stats_dict['Alpha_Monthly']*100:.2f}% (t={stats_dict['Alpha_t']:.2f}, p={stats_dict['Alpha_p']:.4f})")
    print(f"Alpha (Annualized):        {stats_dict['Alpha_Annualized']*100:.2f}%")
    print(f"Information Ratio:         {stats_dict['Information_Ratio']:.3f}")
    print(f"CAPM R^2:                  {stats_dict['CAPM_R2']:.3f}")

    print("\n" + "=" * 70)


def main():
    """
    Main backtest pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting Backtest Analysis")
    logger.info("=" * 60)

    # Load data
    bab_df, benchmark = load_data()

    # Compute all statistics
    stats_dict = compute_all_statistics(bab_df['BAB_Excess_Return'], benchmark)

    # Create monthly performance
    monthly_perf = create_monthly_performance(bab_df, benchmark)

    # Save outputs
    save_outputs(stats_dict, monthly_perf)

    # Print summary
    print_summary(stats_dict)

    logger.info("Backtest analysis complete!")


if __name__ == '__main__':
    main()
