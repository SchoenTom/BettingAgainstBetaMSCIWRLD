# Academic BAB Methodology

## Frazzini & Pedersen (2014) Replication

This document describes the academic implementation of the Betting Against Beta (BAB) strategy following Frazzini & Pedersen (2014), "Betting Against Beta", *Journal of Financial Economics*, 111(1), 1-25.

## Key Differences from Original Implementation

### 1. Dollar Neutrality

**Original (WRONG):**
```
r_BAB = (1/β_L) × r_L - (1/β_H) × r_H

where:
- Long exposure: ~1/0.6 ≈ 1.67 dollars
- Short exposure: ~1/1.4 ≈ 0.71 dollars
- Net position: ~0.96 dollars LONG
```

**Academic (CORRECT):**
```
r_BAB = (1/β_L) × r_L - (1/β_H) × r_H

where both legs are normalized so:
- Long exposure: $1
- Short exposure: $1
- Net position: $0 (market neutral)
```

The beta scaling makes the strategy **beta-neutral** (ex-ante beta = 0), but without dollar normalization, you still have market exposure through the dollar imbalance.

### 2. Beta Estimation

**Original:**
- Simple 60-month rolling OLS regression
- β = Cov(r_i, r_m) / Var(r_m)

**Academic (F&P):**
- Correlation estimated over 1 year (12 months)
- Volatilities estimated over 5 years (60 months)
- Formula: β_TS = ρ(r_i, r_m) × (σ_i / σ_m)
- Shrinkage: β = 0.6 × β_TS + 0.4 × 1.0

The shrinkage toward 1 reduces estimation error for extreme betas.

### 3. Risk-Free Rate

**Original:**
- ^IRX (3-month T-bill from Yahoo Finance)

**Academic:**
- 1-month T-bill from Ken French Data Library
- This is the standard academic source

### 4. Portfolio Weighting

**Original:**
- Equal-weighted portfolios

**Academic:**
- Value-weighted (market cap weighted) portfolios
- Matches academic literature standard

### 5. Winsorization

**Original:**
- No winsorization

**Academic:**
- Extreme returns winsorized at 0.5% tails
- Reduces impact of outliers

### 6. Number of Portfolios

**Original:**
- Quintiles (5 groups)

**Academic:**
- Deciles (10 groups) as in F&P

## Formula Derivation

### BAB Return Formula

The BAB strategy goes long low-beta stocks and short high-beta stocks, with each leg scaled by inverse beta:

```
r_BAB = (1/β_L) × (r_L - r_f) - (1/β_H) × (r_H - r_f)
```

where:
- r_L = return on low-beta portfolio
- r_H = return on high-beta portfolio
- β_L = beta of low-beta portfolio
- β_H = beta of high-beta portfolio
- r_f = risk-free rate

### Ex-Ante Market Beta

After scaling:
- Long leg beta: (1/β_L) × β_L = 1
- Short leg beta: (1/β_H) × β_H = 1
- Net beta: 1 - 1 = 0

This makes BAB market-neutral **by construction**.

### Dollar Position

Without additional normalization:
- Long dollars: 1/β_L (e.g., 1/0.6 = 1.67)
- Short dollars: 1/β_H (e.g., 1/1.4 = 0.71)
- Net dollars: 1/β_L - 1/β_H (e.g., 0.96 LONG)

This creates unintended market exposure!

### True F&P Implementation

F&P normalize so that each leg uses $1:
- Long: $1 invested in low-beta portfolio, levered by 1/β_L
- Short: $1 borrowed in high-beta portfolio, levered by 1/β_H

The leverage is implicit in the strategy, not explicit cash.

## Data Requirements

### Academic Standard (Ideal)

1. **CRSP Monthly Stock File**
   - Point-in-time prices
   - Historical constituents
   - Delisting returns

2. **Compustat**
   - Market capitalization
   - Book values
   - Historical data

3. **Ken French Data Library**
   - Risk-free rate
   - Factor returns
   - Industry classifications

### This Implementation (Free Data)

1. **Yahoo Finance (yfinance)**
   - Stock prices
   - Approximate market cap

2. **Ken French Data Library**
   - 1-month T-bill rate

3. **Wikipedia**
   - S&P 500 historical changes

## Known Limitations

### Survivorship Bias

The most significant limitation. We use:
- Current index constituents applied historically
- Only stocks that survived to today

**Impact:** Overstates returns because failed companies are excluded.

**Mitigation attempts:**
- Track S&P 500 historical changes
- Include known delisted stocks
- Use broader universe

### Market Cap Data

We approximate historical market cap using:
- Current shares outstanding × historical price

**Impact:** Inaccurate value-weighting for historical periods.

### Delisting Returns

We don't have proper delisting returns from CRSP.

**Impact:** Missing the typically negative returns when stocks delist.

## Expected Results

### With Academic Methodology

- Lower absolute returns than original
- More consistent risk-adjusted returns
- Positive alpha relative to CAPM
- Near-zero market beta
- Sharpe ratio ~0.3-0.6

### Why Original Has Higher Returns

1. Net long market position (~$0.96)
2. Bull market from 2017-2025
3. Survivorship bias (only survivors)
4. No delisting returns

## References

1. Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.

2. Black, F. (1972). Capital market equilibrium with restricted borrowing. *The Journal of Business*, 45(3), 444-455.

3. Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance*, 47(2), 427-465.

## Usage

```bash
# Run academic implementation
python main.py --academic

# Compare with original
python main.py --compare

# Run standalone
python bab_academic.py
```

## Output Files

- `output/bab_academic_portfolio.csv` - Monthly BAB returns
- `output/bab_academic_quantiles.csv` - Decile returns
- `data/bab_academic_betas.csv` - Shrunk betas
- `data/ken_french_rf.csv` - 1-month T-bill
- `data/msci_world_benchmark.csv` - Historical benchmark
