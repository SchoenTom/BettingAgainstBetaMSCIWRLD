#!/usr/bin/env python
"""
main.py - Run the complete BAB strategy pipeline

This script executes all components in sequence:
1. data_loader.py - Download and prepare data
2. portfolio_construction.py - Build BAB portfolios
3. backtest.py - Compute performance statistics
4. illustrations.py - Generate visualizations

NEW: Academic mode (--academic) runs Frazzini & Pedersen (2014) methodology:
- Dollar-neutral portfolio construction
- Beta shrinkage (0.6*β + 0.4)
- 1-month T-bill from Ken French
- Value-weighted portfolios
- Proper beta estimation (1Y corr, 5Y vol)

Usage:
    python main.py [--skip-download] [--skip-plots]
    python main.py --academic          # Run F&P methodology
    python main.py --compare           # Compare both versions

Author: BAB Strategy Implementation
"""

import argparse
import sys
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_loader():
    """Run data loading script."""
    logger.info("=" * 70)
    logger.info("STEP 1: Running data_loader.py")
    logger.info("=" * 70)

    from data_loader import main as data_loader_main
    data_loader_main()


def run_portfolio_construction():
    """Run portfolio construction script."""
    logger.info("=" * 70)
    logger.info("STEP 2: Running portfolio_construction.py")
    logger.info("=" * 70)

    from portfolio_construction import main as portfolio_main
    portfolio_main()


def run_backtest():
    """Run backtesting script."""
    logger.info("=" * 70)
    logger.info("STEP 3: Running backtest.py")
    logger.info("=" * 70)

    from backtest import main as backtest_main
    backtest_main()


def run_illustrations():
    """Run illustration generation script."""
    logger.info("=" * 70)
    logger.info("STEP 4: Running illustrations.py")
    logger.info("=" * 70)

    from illustrations import main as illustrations_main
    illustrations_main()


def run_academic_bab():
    """Run academic (Frazzini & Pedersen) BAB implementation."""
    logger.info("=" * 70)
    logger.info("ACADEMIC MODE: Running Frazzini & Pedersen (2014) BAB")
    logger.info("=" * 70)

    from bab_academic import main as academic_main
    return academic_main()


def run_comparison():
    """Run comparison between original and academic implementations."""
    logger.info("=" * 70)
    logger.info("COMPARISON MODE: Original vs Academic BAB")
    logger.info("=" * 70)

    from compare_strategies import main as compare_main
    return compare_main()


def main():
    """
    Main entry point for running the complete BAB pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Run the Betting-Against-Beta (BAB) strategy pipeline'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step (use existing data)'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation step'
    )
    parser.add_argument(
        '--only-download',
        action='store_true',
        help='Only run data download'
    )
    parser.add_argument(
        '--only-backtest',
        action='store_true',
        help='Only run portfolio construction and backtest'
    )
    parser.add_argument(
        '--academic',
        action='store_true',
        help='Run academic (Frazzini & Pedersen 2014) BAB methodology'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare original vs academic implementations'
    )

    args = parser.parse_args()

    start_time = time.time()

    print("\n")
    print("=" * 70)
    print("  BETTING-AGAINST-BETA (BAB) STRATEGY - MSCI WORLD")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.academic:
        print("  MODE: Academic (Frazzini & Pedersen 2014)")
    elif args.compare:
        print("  MODE: Comparison Analysis")
    else:
        print("  MODE: Original Implementation")
    print("=" * 70)
    print("\n")

    try:
        # Academic mode
        if args.academic:
            run_academic_bab()
            elapsed_time = time.time() - start_time
            print("\n")
            print("=" * 70)
            print("  ACADEMIC BAB COMPLETED")
            print("=" * 70)
            print(f"  Total runtime: {elapsed_time:.1f} seconds")
            print("=" * 70)
            return

        # Comparison mode
        if args.compare:
            run_comparison()
            elapsed_time = time.time() - start_time
            print("\n")
            print("=" * 70)
            print("  COMPARISON ANALYSIS COMPLETED")
            print("=" * 70)
            print(f"  Total runtime: {elapsed_time:.1f} seconds")
            print("=" * 70)
            return

        # Step 1: Data Loading (original mode)
        if args.only_download:
            run_data_loader()
            print("\n✓ Data download completed. Exiting as requested.")
            return

        if not args.skip_download:
            run_data_loader()
        else:
            logger.info("Skipping data download (--skip-download flag set)")

        # Step 2: Portfolio Construction
        run_portfolio_construction()

        # Step 3: Backtesting
        run_backtest()

        if args.only_backtest:
            print("\n✓ Backtest completed. Skipping plots as requested.")
        else:
            # Step 4: Illustrations
            if not args.skip_plots:
                run_illustrations()
            else:
                logger.info("Skipping plot generation (--skip-plots flag set)")

        # Summary
        elapsed_time = time.time() - start_time

        print("\n")
        print("=" * 70)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"  Total runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\n")
        print("Next steps:")
        print("  - Review output files in the 'output/' directory")
        print("  - Run 'streamlit run dashboard.py' for interactive analysis")
        print("\n")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
