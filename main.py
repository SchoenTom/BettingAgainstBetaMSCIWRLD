#!/usr/bin/env python
"""
main.py - Run the complete BAB strategy pipeline

This script executes all components in sequence:
1. data_loader.py - Download and prepare data
2. portfolio_construction.py - Build BAB portfolios
3. backtest.py - Compute performance statistics
4. illustrations.py - Generate visualizations

Usage:
    python main.py                    # Run full pipeline
    python main.py --skip-download    # Use existing data
    python main.py --skip-plots       # Skip visualization
    python main.py --only-download    # Just download data
    python main.py --only-backtest    # Skip visualizations

Author: BAB Strategy Implementation
"""

import argparse
import sys
import time
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config
from config import DATA_DIR, OUTPUT_DIR, ensure_directories


def check_data_exists():
    """Check if required data files exist."""
    required_files = [
        os.path.join(DATA_DIR, 'monthly_excess_returns.csv'),
        os.path.join(DATA_DIR, 'rolling_betas.csv'),
    ]
    return all(os.path.exists(f) for f in required_files)


def check_portfolio_exists():
    """Check if portfolio output exists."""
    return os.path.exists(os.path.join(OUTPUT_DIR, 'bab_portfolio.csv'))


def check_backtest_exists():
    """Check if backtest output exists."""
    return os.path.exists(os.path.join(OUTPUT_DIR, 'bab_monthly_performance.csv'))


def run_data_loader():
    """Run data loading script."""
    logger.info("=" * 70)
    logger.info("STEP 1: Data Loading")
    logger.info("=" * 70)

    from data_loader import main as data_loader_main
    data_loader_main()

    # Validate output
    if not check_data_exists():
        raise RuntimeError("Data loader completed but required files are missing!")

    logger.info("Data loading completed successfully")


def run_portfolio_construction():
    """Run portfolio construction script."""
    logger.info("=" * 70)
    logger.info("STEP 2: Portfolio Construction")
    logger.info("=" * 70)

    # Check prerequisites
    if not check_data_exists():
        raise RuntimeError("Data files missing. Run data_loader.py first.")

    from portfolio_construction import main as portfolio_main
    portfolio_main()

    # Validate output
    if not check_portfolio_exists():
        raise RuntimeError("Portfolio construction completed but output is missing!")

    logger.info("Portfolio construction completed successfully")


def run_backtest():
    """Run backtesting script."""
    logger.info("=" * 70)
    logger.info("STEP 3: Backtesting")
    logger.info("=" * 70)

    # Check prerequisites
    if not check_portfolio_exists():
        raise RuntimeError("Portfolio file missing. Run portfolio_construction.py first.")

    from backtest import main as backtest_main
    backtest_main()

    # Validate output
    if not check_backtest_exists():
        raise RuntimeError("Backtest completed but output is missing!")

    logger.info("Backtesting completed successfully")


def run_illustrations():
    """Run illustration generation script."""
    logger.info("=" * 70)
    logger.info("STEP 4: Generating Visualizations")
    logger.info("=" * 70)

    # Check prerequisites
    if not check_backtest_exists():
        raise RuntimeError("Backtest file missing. Run backtest.py first.")

    from illustrations import main as illustrations_main
    illustrations_main()

    logger.info("Visualization generation completed successfully")


def print_banner(mode="full"):
    """Print startup banner."""
    print("\n")
    print("=" * 70)
    print("  BETTING-AGAINST-BETA (BAB) STRATEGY - MSCI WORLD")
    print("  Frazzini & Pedersen (2014) Implementation")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {mode}")
    print("=" * 70)
    print("\n")


def print_summary(elapsed_time, steps_run):
    """Print completion summary."""
    print("\n")
    print("=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Steps executed: {', '.join(steps_run)}")
    print(f"  Total runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n")
    print("Output files:")
    print(f"  - Data: {DATA_DIR}/")
    print(f"  - Results: {OUTPUT_DIR}/")
    print("\n")


def main():
    """Main entry point for running the complete BAB pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the Betting-Against-Beta (BAB) strategy pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full pipeline
  python main.py --skip-download    Use existing data files
  python main.py --skip-plots       Skip visualization generation
  python main.py --only-download    Just download and prepare data
  python main.py --only-backtest    Run through backtest, skip plots
        """
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
        help='Run through backtest, skip plots'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Determine mode
    if args.only_download:
        mode = "Data Download Only"
    elif args.only_backtest:
        mode = "Through Backtest (no plots)"
    elif args.skip_download and args.skip_plots:
        mode = "Portfolio + Backtest Only"
    elif args.skip_download:
        mode = "Skip Download"
    elif args.skip_plots:
        mode = "Skip Plots"
    else:
        mode = "Full Pipeline"

    print_banner(mode)

    start_time = time.time()
    steps_run = []

    try:
        # Step 1: Data Loading
        if args.only_download:
            run_data_loader()
            steps_run.append("Data Loading")
            elapsed_time = time.time() - start_time
            print_summary(elapsed_time, steps_run)
            return

        if not args.skip_download:
            run_data_loader()
            steps_run.append("Data Loading")
        else:
            if not check_data_exists():
                logger.error("--skip-download specified but data files are missing!")
                logger.error(f"Please run without --skip-download or ensure files exist in {DATA_DIR}")
                sys.exit(1)
            logger.info("Skipping data download (--skip-download flag set)")

        # Step 2: Portfolio Construction
        run_portfolio_construction()
        steps_run.append("Portfolio Construction")

        # Step 3: Backtesting
        run_backtest()
        steps_run.append("Backtesting")

        # Step 4: Illustrations (unless skipped)
        if args.only_backtest or args.skip_plots:
            logger.info("Skipping plot generation as requested")
        else:
            run_illustrations()
            steps_run.append("Visualizations")

        # Print summary
        elapsed_time = time.time() - start_time
        print_summary(elapsed_time, steps_run)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.error("Please check that previous pipeline steps completed successfully.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
