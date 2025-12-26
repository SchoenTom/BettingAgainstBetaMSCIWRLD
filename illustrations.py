"""
illustrations.py - Generate visualizations for BAB strategy

This script:
1. Loads backtest output data
2. Generates publication-quality plots:
   - Cumulative equity curves (BAB vs MSCI World)
   - Rolling 12-month Sharpe ratio
   - Beta spread over time
   - Drawdown comparison
   - Quintile return analysis
3. Saves all plots as PNG files

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import OUTPUT_DIR, FIGURE_SIZE, FIGURE_DPI, COLORS, NUM_QUINTILES, ensure_directories

# Try to import seaborn for better styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("seaborn not installed. Using default matplotlib style.")


def setup_plot_style():
    """Setup matplotlib style for publication-quality plots."""
    if HAS_SEABORN:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('seaborn-whitegrid')

    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10


def load_data():
    """
    Load all required data for plotting.

    Returns:
        tuple: (monthly_perf, bab_portfolio, quintile_returns)
    """
    logger.info("Loading data for illustrations...")

    # Load monthly performance
    perf_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    if not os.path.exists(perf_path):
        raise FileNotFoundError(f"Missing {perf_path}. Run backtest.py first.")
    monthly_perf = pd.read_csv(perf_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded monthly performance: {monthly_perf.shape}")

    # Load BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    if not os.path.exists(bab_path):
        raise FileNotFoundError(f"Missing {bab_path}. Run portfolio_construction.py first.")
    bab_portfolio = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded BAB portfolio: {bab_portfolio.shape}")

    # Load quintile returns
    quintile_path = os.path.join(OUTPUT_DIR, 'quintile_returns.csv')
    if not os.path.exists(quintile_path):
        raise FileNotFoundError(f"Missing {quintile_path}. Run portfolio_construction.py first.")
    quintile_returns = pd.read_csv(quintile_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded quintile returns: {quintile_returns.shape}")

    return monthly_perf, bab_portfolio, quintile_returns


def format_date_axis(ax, rotation=45):
    """Format date axis with year labels."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=rotation)


def plot_cumulative_returns(monthly_perf, log_scale=True):
    """Plot cumulative equity curves for BAB and benchmark."""
    logger.info("Creating cumulative returns plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
            color=COLORS['bab'], linewidth=2, label='BAB Strategy')
    ax.plot(monthly_perf.index, monthly_perf['Benchmark_Cumulative'],
            color=COLORS['benchmark'], linewidth=2, label='MSCI World')

    if log_scale:
        ax.set_yscale('log')
        title_suffix = ' (Log Scale)'
    else:
        title_suffix = ''

    ax.set_title(f'Cumulative Returns: BAB Strategy vs MSCI World{title_suffix}', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    format_date_axis(ax)

    plt.tight_layout()
    filename = 'cumulative_returns.png' if log_scale else 'cumulative_returns_linear.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_rolling_sharpe(monthly_perf):
    """Plot rolling 12-month Sharpe ratio."""
    logger.info("Creating rolling Sharpe ratio plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bab_sharpe = monthly_perf['Rolling_12M_BAB_Sharpe'].dropna()
    bench_sharpe = monthly_perf['Rolling_12M_Benchmark_Sharpe'].dropna()

    ax.plot(bab_sharpe.index, bab_sharpe,
            color=COLORS['bab'], linewidth=1.5, label='BAB Strategy')
    ax.plot(bench_sharpe.index, bench_sharpe,
            color=COLORS['benchmark'], linewidth=1.5, label='MSCI World')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    ax.set_title('Rolling 12-Month Sharpe Ratio', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    # Reasonable y-axis limits
    all_vals = pd.concat([bab_sharpe, bench_sharpe]).dropna()
    if len(all_vals) > 0:
        ax.set_ylim([max(-4, all_vals.min() - 0.5), min(4, all_vals.max() + 0.5)])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rolling_sharpe.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_beta_spread(bab_portfolio):
    """Plot beta spread (Q5 - Q1 mean beta) over time."""
    logger.info("Creating beta spread plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    spread = bab_portfolio['Beta_Spread']
    ax.plot(spread.index, spread, color=COLORS['spread'], linewidth=1.5, label='Beta Spread')

    # Rolling mean
    rolling_spread = spread.rolling(12).mean()
    ax.plot(rolling_spread.index, rolling_spread,
            color='darkred', linewidth=2, linestyle='--', label='12-Month Rolling Mean')

    # Average line
    avg_spread = spread.mean()
    ax.axhline(y=avg_spread, color='gray', linestyle=':', alpha=0.7)

    ax.set_title('Beta Spread (Q5 High Beta - Q1 Low Beta)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta Spread')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'beta_spread.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_drawdowns(monthly_perf):
    """Plot drawdown comparison."""
    logger.info("Creating drawdown plot...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.fill_between(monthly_perf.index, monthly_perf['BAB_Drawdown'] * 100, 0,
                    color=COLORS['bab'], alpha=0.5, label='BAB Strategy')
    ax.fill_between(monthly_perf.index, monthly_perf['Benchmark_Drawdown'] * 100, 0,
                    color=COLORS['benchmark'], alpha=0.5, label='MSCI World')

    ax.set_title('Drawdown Analysis', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'drawdowns.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_quintile_returns(quintile_returns):
    """Plot average returns by beta quintile."""
    logger.info("Creating quintile returns plot...")

    # Calculate average returns
    return_cols = [f'Q{i}_Return' for i in range(1, NUM_QUINTILES + 1)]
    avg_returns = quintile_returns[return_cols].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    bars = ax.bar(range(1, NUM_QUINTILES + 1), avg_returns.values,
                  color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Average Monthly Excess Returns by Beta Quintile', fontweight='bold')
    ax.set_xlabel('Beta Quintile (Q1=Low Beta, Q5=High Beta)')
    ax.set_ylabel('Average Monthly Excess Return (%)')
    ax.set_xticks(range(1, NUM_QUINTILES + 1))
    ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, avg_returns.values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'quintile_returns.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_quintile_betas(quintile_returns):
    """Plot average betas by quintile."""
    logger.info("Creating quintile betas plot...")

    beta_cols = [f'Q{i}_Mean_Beta' for i in range(1, NUM_QUINTILES + 1)]
    avg_betas = quintile_returns[beta_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    bars = ax.bar(range(1, NUM_QUINTILES + 1), avg_betas.values,
                  color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Average Beta by Quintile', fontweight='bold')
    ax.set_xlabel('Beta Quintile')
    ax.set_ylabel('Average Beta')
    ax.set_xticks(range(1, NUM_QUINTILES + 1))
    ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Market Beta = 1')
    ax.legend(loc='upper left')

    # Value labels
    for bar, val in zip(bars, avg_betas.values):
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'quintile_betas.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_return_distribution(monthly_perf):
    """Plot return distribution comparison."""
    logger.info("Creating return distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bab_ret = monthly_perf['BAB_Return'] * 100
    bench_ret = monthly_perf['Benchmark_Return'] * 100

    # Histogram
    ax1.hist(bab_ret, bins=25, alpha=0.7, color=COLORS['bab'],
             label='BAB Strategy', edgecolor='black')
    ax1.hist(bench_ret, bins=25, alpha=0.7, color=COLORS['benchmark'],
             label='MSCI World', edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Monthly Return Distribution', fontweight='bold')
    ax1.set_xlabel('Monthly Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Box plot
    data = [bab_ret.dropna(), bench_ret.dropna()]
    bp = ax2.boxplot(data, labels=['BAB Strategy', 'MSCI World'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['bab'])
    bp['boxes'][1].set_facecolor(COLORS['benchmark'])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Return Distribution (Box Plot)', fontweight='bold')
    ax2.set_ylabel('Monthly Return (%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'return_distribution.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def create_summary_dashboard(monthly_perf, bab_portfolio, quintile_returns):
    """Create a multi-panel summary dashboard."""
    logger.info("Creating summary dashboard...")

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Cumulative returns (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
             color=COLORS['bab'], linewidth=2, label='BAB Strategy')
    ax1.plot(monthly_perf.index, monthly_perf['Benchmark_Cumulative'],
             color=COLORS['benchmark'], linewidth=2, label='MSCI World')
    ax1.set_title('Cumulative Returns', fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    format_date_axis(ax1)

    # Panel 2: Rolling Sharpe (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    bab_sharpe = monthly_perf['Rolling_12M_BAB_Sharpe'].dropna()
    bench_sharpe = monthly_perf['Rolling_12M_Benchmark_Sharpe'].dropna()
    ax2.plot(bab_sharpe.index, bab_sharpe,
             color=COLORS['bab'], linewidth=1.5, label='BAB')
    ax2.plot(bench_sharpe.index, bench_sharpe,
             color=COLORS['benchmark'], linewidth=1.5, label='MSCI World')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Rolling 12-Month Sharpe Ratio', fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    format_date_axis(ax2)

    # Panel 3: Drawdowns (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.fill_between(monthly_perf.index, monthly_perf['BAB_Drawdown'] * 100, 0,
                     color=COLORS['bab'], alpha=0.5, label='BAB')
    ax3.fill_between(monthly_perf.index, monthly_perf['Benchmark_Drawdown'] * 100, 0,
                     color=COLORS['benchmark'], alpha=0.5, label='MSCI World')
    ax3.set_title('Drawdowns', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    format_date_axis(ax3)

    # Panel 4: Quintile returns (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    return_cols = [f'Q{i}_Return' for i in range(1, NUM_QUINTILES + 1)]
    avg_returns = quintile_returns[return_cols].mean() * 100
    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    ax4.bar(range(1, NUM_QUINTILES + 1), avg_returns.values, color=colors, edgecolor='black')
    ax4.set_title('Avg Monthly Return by Beta Quintile', fontweight='bold')
    ax4.set_xlabel('Quintile (Q1=Low, Q5=High Beta)')
    ax4.set_ylabel('Avg Monthly Return (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'summary_dashboard.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    """Main illustration generation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Illustration Generation")
    logger.info("=" * 60)

    ensure_directories()
    setup_plot_style()

    # Load data
    try:
        monthly_perf, bab_portfolio, quintile_returns = load_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}")
        print("Please run the full pipeline first:")
        print("  1. python data_loader.py")
        print("  2. python portfolio_construction.py")
        print("  3. python backtest.py")
        print("  4. python illustrations.py")
        return

    # Generate all plots
    plot_cumulative_returns(monthly_perf, log_scale=True)
    plot_cumulative_returns(monthly_perf, log_scale=False)
    plot_rolling_sharpe(monthly_perf)
    plot_beta_spread(bab_portfolio)
    plot_drawdowns(monthly_perf)
    plot_quintile_returns(quintile_returns)
    plot_quintile_betas(quintile_returns)
    plot_return_distribution(monthly_perf)
    create_summary_dashboard(monthly_perf, bab_portfolio, quintile_returns)

    logger.info("=" * 60)
    logger.info("All illustrations generated successfully!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    print(f"\nGenerated plots saved to: {OUTPUT_DIR}")
    print("\nPlots generated:")
    print("  - cumulative_returns.png (log scale)")
    print("  - cumulative_returns_linear.png")
    print("  - rolling_sharpe.png")
    print("  - beta_spread.png")
    print("  - drawdowns.png")
    print("  - quintile_returns.png")
    print("  - quintile_betas.png")
    print("  - return_distribution.png")
    print("  - summary_dashboard.png")


if __name__ == '__main__':
    main()
