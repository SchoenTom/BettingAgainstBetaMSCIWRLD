"""
illustrations.py - Generate visualizations for BAB strategy

This script:
1. Loads backtest output data
2. Generates publication-quality plots:
   - Cumulative equity curves (BAB vs MSCI World)
   - Rolling 12-month Sharpe ratio
   - Beta spread (Q5 - Q1 mean beta)
   - Drawdown comparison
   - Quintile return comparison
3. Saves all plots as PNG files

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging

try:
    import seaborn as sns
except ModuleNotFoundError:
    print("seaborn not installed; please `pip install seaborn` to generate charts. Skipping illustration run.")
    raise SystemExit(0)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Plot style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'bab': '#1f77b4',       # Blue
    'benchmark': '#ff7f0e',  # Orange
    'q1': '#2ca02c',        # Green
    'q5': '#d62728',        # Red
    'spread': '#9467bd',    # Purple
    'neutral': '#7f7f7f',   # Gray
}
FIG_SIZE = (12, 6)
DPI = 150


def load_data():
    """
    Load all required data for plotting.

    Returns:
        tuple: (monthly_perf, bab_portfolio, quintile_returns)
    """
    logger.info("Loading data for illustrations...")

    # Load monthly performance
    perf_path = os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv')
    monthly_perf = pd.read_csv(perf_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded monthly performance: {monthly_perf.shape}")

    # Load BAB portfolio
    bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
    bab_portfolio = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded BAB portfolio: {bab_portfolio.shape}")

    # Load quintile returns
    quintile_path = os.path.join(OUTPUT_DIR, 'quintile_returns.csv')
    quintile_returns = pd.read_csv(quintile_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded quintile returns: {quintile_returns.shape}")

    return monthly_perf, bab_portfolio, quintile_returns


def setup_plot():
    """
    Setup common plot parameters.
    """
    plt.rcParams['figure.figsize'] = FIG_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def format_date_axis(ax, rotation=45):
    """
    Format date axis with appropriate locator and formatter.

    Args:
        ax: Matplotlib axis object
        rotation: X-tick label rotation
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=rotation)
    # ensure full span is shown
    ax.set_xlim(left=None, right=None)


def plot_cumulative_returns(monthly_perf):
    """
    Plot cumulative equity curves for BAB and benchmark.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating cumulative returns plot...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
            color=COLORS['bab'], linewidth=2, label='BAB Strategy')
    ax.plot(monthly_perf.index, monthly_perf['Benchmark_Cumulative'],
            color=COLORS['benchmark'], linewidth=2, label='MSCI World')

    ax.set_title('Cumulative Returns: BAB Strategy vs MSCI World', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(monthly_perf.index.min(), monthly_perf.index.max())
    format_date_axis(ax)

    # Add horizontal line at $1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cumulative_returns.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_cumulative_returns_linear(monthly_perf):
    """
    Plot cumulative equity curves (linear scale).

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating cumulative returns plot (linear)...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(monthly_perf.index, monthly_perf['BAB_Cumulative'],
            color=COLORS['bab'], linewidth=2, label='BAB Strategy')
    ax.plot(monthly_perf.index, monthly_perf['Benchmark_Cumulative'],
            color=COLORS['benchmark'], linewidth=2, label='MSCI World')

    ax.set_title('Cumulative Returns: BAB Strategy vs MSCI World (Linear Scale)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(monthly_perf.index.min(), monthly_perf.index.max())
    format_date_axis(ax)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cumulative_returns_linear.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_rolling_sharpe(monthly_perf):
    """
    Plot rolling 12-month Sharpe ratio.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating rolling Sharpe ratio plot...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(monthly_perf.index, monthly_perf['Rolling_12M_BAB_Sharpe'],
            color=COLORS['bab'], linewidth=1.5, label='BAB Strategy')
    ax.plot(monthly_perf.index, monthly_perf['Rolling_12M_Benchmark_Sharpe'],
            color=COLORS['benchmark'], linewidth=1.5, label='MSCI World')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    ax.set_title('Rolling 12-Month Sharpe Ratio', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    # Set reasonable y-axis limits
    y_min = min(monthly_perf['Rolling_12M_BAB_Sharpe'].min(),
                monthly_perf['Rolling_12M_Benchmark_Sharpe'].min())
    y_max = max(monthly_perf['Rolling_12M_BAB_Sharpe'].max(),
                monthly_perf['Rolling_12M_Benchmark_Sharpe'].max())
    ax.set_ylim([max(-3, y_min - 0.5), min(3, y_max + 0.5)])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rolling_sharpe.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_beta_spread(bab_portfolio):
    """
    Plot beta spread (Q5 mean beta - Q1 mean beta) over time.

    Args:
        bab_portfolio: DataFrame with BAB portfolio data
    """
    logger.info("Creating beta spread plot...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(bab_portfolio.index, bab_portfolio['Beta_Spread'],
            color=COLORS['spread'], linewidth=1.5)

    # Add rolling mean
    rolling_spread = bab_portfolio['Beta_Spread'].rolling(12).mean()
    ax.plot(bab_portfolio.index, rolling_spread,
            color='darkred', linewidth=2, linestyle='--',
            label='12-Month Rolling Mean')

    ax.set_title('Beta Spread (Q5 High Beta - Q1 Low Beta)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta Spread')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    # Add average line
    avg_spread = bab_portfolio['Beta_Spread'].mean()
    ax.axhline(y=avg_spread, color='gray', linestyle=':', alpha=0.7,
               label=f'Average: {avg_spread:.2f}')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'beta_spread.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_drawdowns(monthly_perf):
    """
    Plot drawdown comparison.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating drawdown plot...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

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
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_quintile_returns(quintile_returns):
    """
    Plot average returns by beta quintile.

    Args:
        quintile_returns: DataFrame with quintile return data
    """
    logger.info("Creating quintile returns plot...")

    # Calculate average monthly returns for each quintile
    return_cols = [f'Q{i}_Return' for i in range(1, 6)]
    avg_returns = quintile_returns[return_cols].mean() * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    bars = ax.bar(range(1, 6), avg_returns.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Average Monthly Returns by Beta Quintile', fontweight='bold')
    ax.set_xlabel('Beta Quintile (Q1=Low Beta, Q5=High Beta)')
    ax.set_ylabel('Average Monthly Return (%)')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['Q1\n(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Beta)'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add value labels on bars
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
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_quintile_betas(quintile_returns):
    """
    Plot average betas by quintile.

    Args:
        quintile_returns: DataFrame with quintile data
    """
    logger.info("Creating quintile betas plot...")

    # Calculate average betas for each quintile
    beta_cols = [f'Q{i}_Mean_Beta' for i in range(1, 6)]
    avg_betas = quintile_returns[beta_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    bars = ax.bar(range(1, 6), avg_betas.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_title('Average Beta by Quintile', fontweight='bold')
    ax.set_xlabel('Beta Quintile')
    ax.set_ylabel('Average Beta')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(['Q1\n(Low Beta)', 'Q2', 'Q3', 'Q4', 'Q5\n(High Beta)'])
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line at beta = 1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Market Beta = 1')
    ax.legend(loc='upper left')

    # Add value labels on bars
    for bar, val in zip(bars, avg_betas.values):
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'quintile_betas.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_excess_returns(monthly_perf):
    """
    Plot BAB excess returns over benchmark.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating excess returns plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Monthly excess returns
    excess_ret = monthly_perf['Excess_Return'] * 100
    colors = ['green' if x > 0 else 'red' for x in excess_ret]
    ax1.bar(monthly_perf.index, excess_ret, color=colors, alpha=0.7, width=20)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Monthly Excess Returns (BAB - MSCI World)', fontweight='bold')
    ax1.set_ylabel('Excess Return (%)')
    ax1.grid(True, alpha=0.3)

    # Bottom: Cumulative excess
    ax2.plot(monthly_perf.index, monthly_perf['Excess_Cumulative'],
             color=COLORS['spread'], linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Cumulative Excess Returns', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Growth of $1 (Excess)')
    ax2.grid(True, alpha=0.3)
    format_date_axis(ax2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'excess_returns.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_return_distribution(monthly_perf):
    """
    Plot return distribution comparison.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating return distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(monthly_perf['BAB_Return'] * 100, bins=30, alpha=0.7,
             color=COLORS['bab'], label='BAB Strategy', edgecolor='black')
    ax1.hist(monthly_perf['Benchmark_Return'] * 100, bins=30, alpha=0.7,
             color=COLORS['benchmark'], label='MSCI World', edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Monthly Return Distribution', fontweight='bold')
    ax1.set_xlabel('Monthly Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Box plot
    data_for_box = [monthly_perf['BAB_Return'] * 100,
                    monthly_perf['Benchmark_Return'] * 100]
    bp = ax2.boxplot(data_for_box, labels=['BAB Strategy', 'MSCI World'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['bab'])
    bp['boxes'][1].set_facecolor(COLORS['benchmark'])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Return Distribution (Box Plot)', fontweight='bold')
    ax2.set_ylabel('Monthly Return (%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'return_distribution.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_rolling_bab_excess(monthly_perf):
    """
    Plot rolling 12-month BAB excess return over benchmark.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating rolling BAB excess plot...")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Calculate rolling 12-month excess return
    rolling_excess = monthly_perf['Excess_Return'].rolling(12).mean() * 12 * 100

    ax.plot(monthly_perf.index, rolling_excess,
            color=COLORS['spread'], linewidth=2)
    ax.fill_between(monthly_perf.index, rolling_excess, 0,
                    where=(rolling_excess >= 0), color='green', alpha=0.3)
    ax.fill_between(monthly_perf.index, rolling_excess, 0,
                    where=(rolling_excess < 0), color='red', alpha=0.3)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title('Rolling 12-Month BAB Excess Return Over MSCI World (Annualized)',
                 fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Excess Return (%)')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rolling_excess_return.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_yearly_returns(monthly_perf):
    """
    Plot yearly return comparison.

    Args:
        monthly_perf: DataFrame with performance data
    """
    logger.info("Creating yearly returns plot...")

    # Resample to yearly returns
    yearly_bab = (1 + monthly_perf['BAB_Return']).resample('YE').prod() - 1
    yearly_bench = (1 + monthly_perf['Benchmark_Return']).resample('YE').prod() - 1

    years = yearly_bab.index.year

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(years))
    width = 0.35

    bars1 = ax.bar(x - width/2, yearly_bab.values * 100, width,
                   label='BAB Strategy', color=COLORS['bab'], edgecolor='black')
    bars2 = ax.bar(x + width/2, yearly_bench.values * 100, width,
                   label='MSCI World', color=COLORS['benchmark'], edgecolor='black')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Annual Returns: BAB Strategy vs MSCI World', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Return (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'yearly_returns.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def create_summary_dashboard(monthly_perf, bab_portfolio, quintile_returns):
    """
    Create a multi-panel summary dashboard.

    Args:
        monthly_perf: DataFrame with performance data
        bab_portfolio: DataFrame with BAB portfolio data
        quintile_returns: DataFrame with quintile returns
    """
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
    ax2.plot(monthly_perf.index, monthly_perf['Rolling_12M_BAB_Sharpe'],
             color=COLORS['bab'], linewidth=1.5, label='BAB')
    ax2.plot(monthly_perf.index, monthly_perf['Rolling_12M_Benchmark_Sharpe'],
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
    return_cols = [f'Q{i}_Return' for i in range(1, 6)]
    avg_returns = quintile_returns[return_cols].mean() * 100
    colors = [COLORS['q1'], '#5aa02c', '#b8860b', '#d63a28', COLORS['q5']]
    bars = ax4.bar(range(1, 6), avg_returns.values, color=colors, edgecolor='black')
    ax4.set_title('Avg Monthly Return by Beta Quintile', fontweight='bold')
    ax4.set_xlabel('Quintile (Q1=Low, Q5=High Beta)')
    ax4.set_ylabel('Avg Monthly Return (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'summary_dashboard.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    """
    Main illustration generation pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting Illustration Generation")
    logger.info("=" * 60)

    # Setup plot parameters
    setup_plot()

    # Load data
    monthly_perf, bab_portfolio, quintile_returns = load_data()

    # Generate all plots
    plot_cumulative_returns(monthly_perf)
    plot_cumulative_returns_linear(monthly_perf)
    plot_rolling_sharpe(monthly_perf)
    plot_beta_spread(bab_portfolio)
    plot_drawdowns(monthly_perf)
    plot_quintile_returns(quintile_returns)
    plot_quintile_betas(quintile_returns)
    plot_excess_returns(monthly_perf)
    plot_return_distribution(monthly_perf)
    plot_rolling_bab_excess(monthly_perf)
    plot_yearly_returns(monthly_perf)
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
    print("  - excess_returns.png")
    print("  - return_distribution.png")
    print("  - rolling_excess_return.png")
    print("  - yearly_returns.png")
    print("  - summary_dashboard.png")


if __name__ == '__main__':
    main()
