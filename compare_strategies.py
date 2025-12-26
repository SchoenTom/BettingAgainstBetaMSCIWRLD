"""
compare_strategies.py - Compare Original vs Academic BAB Implementation

This script runs both implementations and shows the differences,
highlighting what changes when using proper F&P methodology.

Author: BAB Strategy Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_original_results():
    """Load results from original implementation."""
    try:
        bab_path = os.path.join(OUTPUT_DIR, 'bab_portfolio.csv')
        if os.path.exists(bab_path):
            return pd.read_csv(bab_path, index_col=0, parse_dates=True)
    except Exception as e:
        logger.warning(f"Could not load original results: {e}")
    return None


def load_academic_results():
    """Load results from academic implementation."""
    try:
        bab_path = os.path.join(OUTPUT_DIR, 'bab_academic_portfolio.csv')
        if os.path.exists(bab_path):
            return pd.read_csv(bab_path, index_col=0, parse_dates=True)
    except Exception as e:
        logger.warning(f"Could not load academic results: {e}")
    return None


def compute_statistics(returns: pd.Series, name: str) -> dict:
    """Compute comprehensive statistics for a return series."""
    r = returns.dropna()
    n = len(r)

    if n < 12:
        return {'Name': name, 'N': n, 'Error': 'Insufficient data'}

    mean_m = r.mean()
    std_m = r.std()
    t_stat = mean_m / (std_m / np.sqrt(n)) if std_m > 0 else 0

    # Annualized
    ann_ret = mean_m * 12
    ann_vol = std_m * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Drawdown
    cum = (1 + r).cumprod()
    rolling_max = cum.expanding().max()
    dd = cum / rolling_max - 1
    max_dd = dd.min()

    # Win rate
    win_rate = (r > 0).mean()

    # Skewness and kurtosis
    skew = r.skew()
    kurt = r.kurtosis()

    return {
        'Name': name,
        'N_Months': n,
        'Mean_Monthly_%': mean_m * 100,
        'Std_Monthly_%': std_m * 100,
        'T_Statistic': t_stat,
        'Ann_Return_%': ann_ret * 100,
        'Ann_Vol_%': ann_vol * 100,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown_%': max_dd * 100,
        'Win_Rate_%': win_rate * 100,
        'Skewness': skew,
        'Kurtosis': kurt
    }


def plot_comparison(original: pd.DataFrame, academic: pd.DataFrame):
    """Create comparison plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get return columns
    orig_ret = original['BAB_Return'] if 'BAB_Return' in original.columns else original.iloc[:, 0]
    acad_ret = academic['BAB_Return'] if 'BAB_Return' in academic.columns else academic.iloc[:, 0]

    # Align dates
    common_dates = orig_ret.index.intersection(acad_ret.index)
    orig_ret = orig_ret.loc[common_dates]
    acad_ret = acad_ret.loc[common_dates]

    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    orig_cum = (1 + orig_ret).cumprod()
    acad_cum = (1 + acad_ret).cumprod()

    ax1.plot(orig_cum.index, orig_cum, label='Original (Non-Dollar-Neutral)', color='red', linewidth=2)
    ax1.plot(acad_cum.index, acad_cum, label='Academic (Dollar-Neutral)', color='blue', linewidth=2)
    ax1.set_title('Cumulative Returns Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Growth of $1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Monthly Returns Distribution
    ax2 = axes[0, 1]
    ax2.hist(orig_ret * 100, bins=30, alpha=0.6, label='Original', color='red', edgecolor='black')
    ax2.hist(acad_ret * 100, bins=30, alpha=0.6, label='Academic', color='blue', edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.set_title('Monthly Return Distribution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Monthly Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rolling 12-Month Sharpe
    ax3 = axes[1, 0]
    orig_sharpe = (orig_ret.rolling(12).mean() * 12) / (orig_ret.rolling(12).std() * np.sqrt(12))
    acad_sharpe = (acad_ret.rolling(12).mean() * 12) / (acad_ret.rolling(12).std() * np.sqrt(12))

    ax3.plot(orig_sharpe.index, orig_sharpe, label='Original', color='red', linewidth=1.5)
    ax3.plot(acad_sharpe.index, acad_sharpe, label='Academic', color='blue', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_title('Rolling 12-Month Sharpe Ratio', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Net Dollar Position (if available)
    ax4 = axes[1, 1]
    if 'Net_Dollars' in academic.columns:
        net_dollars = academic.loc[common_dates, 'Net_Dollars']
        ax4.plot(net_dollars.index, net_dollars, label='Academic Net $', color='blue', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Dollar Neutral (Target)')
        ax4.set_title('Net Dollar Position (Academic)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Net $ Position')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Show drawdown comparison instead
        orig_cum = (1 + orig_ret).cumprod()
        acad_cum = (1 + acad_ret).cumprod()
        orig_dd = (orig_cum / orig_cum.expanding().max() - 1) * 100
        acad_dd = (acad_cum / acad_cum.expanding().max() - 1) * 100

        ax4.fill_between(orig_dd.index, orig_dd, 0, alpha=0.5, color='red', label='Original')
        ax4.fill_between(acad_dd.index, acad_dd, 0, alpha=0.5, color='blue', label='Academic')
        ax4.set_title('Drawdown Comparison', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'strategy_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot to {save_path}")


def print_comparison_table(orig_stats: dict, acad_stats: dict):
    """Print side-by-side comparison table."""
    print("\n" + "="*80)
    print("  STRATEGY COMPARISON: ORIGINAL vs ACADEMIC (F&P) IMPLEMENTATION")
    print("="*80)

    print("\n" + "-"*80)
    print(f"{'Metric':<30} {'Original':>20} {'Academic (F&P)':>20}")
    print("-"*80)

    metrics = [
        ('N_Months', 'Months', '{:.0f}'),
        ('Mean_Monthly_%', 'Mean Monthly (%)', '{:.3f}'),
        ('Std_Monthly_%', 'Std Monthly (%)', '{:.3f}'),
        ('T_Statistic', 'T-Statistic', '{:.2f}'),
        ('Ann_Return_%', 'Annualized Return (%)', '{:.2f}'),
        ('Ann_Vol_%', 'Annualized Vol (%)', '{:.2f}'),
        ('Sharpe_Ratio', 'Sharpe Ratio', '{:.3f}'),
        ('Max_Drawdown_%', 'Max Drawdown (%)', '{:.2f}'),
        ('Win_Rate_%', 'Win Rate (%)', '{:.1f}'),
        ('Skewness', 'Skewness', '{:.3f}'),
        ('Kurtosis', 'Kurtosis', '{:.3f}'),
    ]

    for key, label, fmt in metrics:
        orig_val = orig_stats.get(key, 'N/A')
        acad_val = acad_stats.get(key, 'N/A')

        orig_str = fmt.format(orig_val) if isinstance(orig_val, (int, float)) else str(orig_val)
        acad_str = fmt.format(acad_val) if isinstance(acad_val, (int, float)) else str(acad_val)

        print(f"{label:<30} {orig_str:>20} {acad_str:>20}")

    print("-"*80)

    # Highlight key differences
    print("\n" + "="*80)
    print("  KEY DIFFERENCES EXPLAINED")
    print("="*80)

    print("""
1. DOLLAR NEUTRALITY:
   - Original: Long$(~1.67) ≠ Short$(~0.71) → Net Long Market Exposure
   - Academic: Long$($1) = Short$($1) → True Market Neutrality

2. BETA ESTIMATION:
   - Original: Simple rolling 60-month OLS beta
   - Academic: F&P formula (1Y correlation × volatility ratio) + shrinkage

3. EXPECTED RESULTS:
   - Academic version should have LOWER but MORE CONSISTENT returns
   - Academic version represents true BAB anomaly (alpha)
   - Original version captures BAB anomaly + market beta exposure

4. WHY ORIGINAL HAS HIGHER RETURNS:
   - 2017-2025 was predominantly a bull market
   - Net long exposure (~0.96$) amplified returns
   - This is NOT arbitrage - it's leveraged market exposure

5. STATISTICAL SIGNIFICANCE:
   - Look at T-statistic: Academic version shows true alpha significance
   - High T-stat in original partially driven by market returns
""")


def main():
    """Run comparison analysis."""
    print("\n" + "="*80)
    print("  BAB STRATEGY COMPARISON ANALYSIS")
    print("="*80 + "\n")

    # Load results
    original = load_original_results()
    academic = load_academic_results()

    if original is None and academic is None:
        print("No results files found. Run the strategies first:")
        print("  1. python main.py (for original)")
        print("  2. python bab_academic.py (for academic)")
        return

    results = {}

    if original is not None:
        logger.info("Loaded original results")
        ret_col = 'BAB_Return' if 'BAB_Return' in original.columns else 'BAB_Excess_Return'
        results['original'] = compute_statistics(original[ret_col], 'Original')
    else:
        results['original'] = {'Name': 'Original', 'Error': 'Not available'}

    if academic is not None:
        logger.info("Loaded academic results")
        ret_col = 'BAB_Return' if 'BAB_Return' in academic.columns else 'BAB_Excess_Return'
        results['academic'] = compute_statistics(academic[ret_col], 'Academic (F&P)')
    else:
        results['academic'] = {'Name': 'Academic', 'Error': 'Not available'}

    # Print comparison
    print_comparison_table(results['original'], results['academic'])

    # Create plots if both available
    if original is not None and academic is not None:
        plot_comparison(original, academic)
        print(f"\nComparison plot saved to: {OUTPUT_DIR}/strategy_comparison.png")

    # Save comparison CSV
    comparison_df = pd.DataFrame([results['original'], results['academic']])
    comparison_path = os.path.join(OUTPUT_DIR, 'strategy_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison data saved to: {comparison_path}")

    return results


if __name__ == '__main__':
    results = main()
