# Betting-Against-Beta (BAB) Strategy on MSCI World

A comprehensive Python implementation of the Betting-Against-Beta (BAB) strategy, originally documented by Frazzini and Pedersen (2014), applied to MSCI World constituents.

## Overview

The BAB strategy exploits the empirical observation that low-beta stocks tend to outperform high-beta stocks on a risk-adjusted basis. This implementation:

- Uses monthly rebalancing with 60-month rolling betas
- Sorts stocks into beta quintiles (Q1 = lowest beta, Q5 = highest beta)
- Constructs BAB return as: Q1 (low beta) - Q5 (high beta)
- Provides comprehensive backtesting and visualization

## Project Structure

```
BettingAgainstBetaMSCIWRLD/
├── data_loader.py           # Download and prepare market data
├── portfolio_construction.py # Build BAB portfolios
├── backtest.py              # Compute performance statistics
├── illustrations.py         # Generate static plots
├── dashboard.py             # Interactive Streamlit dashboard
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Output: CSV data files
│   ├── monthly_prices.csv
│   ├── monthly_returns.csv
│   ├── monthly_excess_returns.csv
│   ├── risk_free_rate.csv
│   └── rolling_betas.csv
└── output/                 # Output: results and plots
    ├── bab_portfolio.csv
    ├── quintile_returns.csv
    ├── bab_backtest_summary.csv
    ├── bab_monthly_performance.csv
    └── *.png (visualization files)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BettingAgainstBetaMSCIWRLD
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in sequence:

### 1. Data Loading
```bash
python data_loader.py
```
Downloads market data from yfinance:
- MSCI World constituent prices (represented by major developed market stocks)
- MSCI World benchmark proxy (URTH ETF)
- Risk-free rate (3-month T-bill, ^IRX)

Computes and saves:
- Monthly prices and returns
- Excess returns (return - risk-free rate)
- 60-month rolling betas

### 2. Portfolio Construction
```bash
python portfolio_construction.py
```
- Loads saved data
- Sorts stocks into 5 beta quintiles each month
- Computes BAB returns (Q1 - Q5)
- Saves portfolio and quintile statistics

### 3. Backtesting
```bash
python backtest.py
```
Computes performance metrics:
- Annualized return and volatility
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown
- Alpha and Information Ratio vs benchmark

### 4. Generate Plots
```bash
python illustrations.py
```
Creates publication-quality visualizations:
- Cumulative equity curves
- Rolling Sharpe ratios
- Beta spread over time
- Drawdown analysis
- Quintile return/beta charts
- Return distributions

### 5. Interactive Dashboard
```bash
streamlit run dashboard.py
```
Opens a web-based interactive dashboard with:
- Filterable date range
- Interactive charts (Plotly)
- Real-time metric calculations
- Quintile analysis tools

## Methodology

### Beta Calculation
Rolling 60-month betas computed as:
```
Beta_i = Cov(R_i - R_f, R_m - R_f) / Var(R_m - R_f)
```
Where:
- R_i = Stock return
- R_m = Market (MSCI World) return
- R_f = Risk-free rate

### Portfolio Formation (No Look-Ahead)
Each month t:
1. Use betas from month t-1 (lagged)
2. Sort stocks into quintiles based on lagged beta
3. Q1 = bottom 20% (low beta), Q5 = top 20% (high beta)
4. BAB_Return_t = EqualWeight(Q1_Returns_t) - EqualWeight(Q5_Returns_t)

### Data Sources
All data from **yfinance**:
- **Stocks**: Major MSCI World constituents (US, Europe, Japan, etc.)
- **Benchmark**: URTH (iShares MSCI World ETF)
- **Risk-Free**: ^IRX (3-month T-bill rate)

## Key Outputs

### CSV Files
| File | Description |
|------|-------------|
| `monthly_prices.csv` | Monthly adjusted close prices |
| `monthly_returns.csv` | Simple monthly returns |
| `monthly_excess_returns.csv` | Returns minus risk-free rate |
| `rolling_betas.csv` | 60-month rolling betas |
| `bab_portfolio.csv` | BAB returns and quintile stats |
| `quintile_returns.csv` | Returns for all quintiles |
| `bab_backtest_summary.csv` | Performance summary |
| `bab_monthly_performance.csv` | Full monthly performance |

### Visualizations
| Plot | Description |
|------|-------------|
| `cumulative_returns.png` | BAB vs MSCI World (log scale) |
| `rolling_sharpe.png` | 12-month rolling Sharpe |
| `beta_spread.png` | Q5 - Q1 beta spread |
| `drawdowns.png` | Drawdown comparison |
| `quintile_returns.png` | Average returns by quintile |
| `summary_dashboard.png` | Multi-panel summary |

## Limitations & Notes

1. **Survivorship Bias**: Uses current constituents applied historically (intentional simplification per spec)
2. **URTH Inception**: URTH ETF started in 2012; earlier benchmark data limited
3. **No Leverage**: Simple Q1 - Q5 construction without dollar-neutral scaling
4. **Transaction Costs**: Not modeled in backtest

## References

- Frazzini, A., & Pedersen, L. H. (2014). "Betting against beta." *Journal of Financial Economics*, 111(1), 1-25.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
