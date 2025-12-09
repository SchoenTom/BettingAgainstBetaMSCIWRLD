"""
factor_regressions.py - Simple CAPM regression diagnostics for BAB excess returns

This module runs a CAPM regression of BAB EXCESS returns on URTH EXCESS (proxy
for MSCI World) to verify ex-post market neutrality. It assumes the pipeline has
already produced:
- output/bab_portfolio.csv   (contains BAB_Excess_Return)
- data/monthly_excess_returns.csv (contains MSCI_World excess)

All returns are treated as excess; alpha is monthly and annualized (×12).
"""

import math
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_excess_series():
    bab_path = os.path.join(OUTPUT_DIR, "bab_portfolio.csv")
    excess_path = os.path.join(DATA_DIR, "monthly_excess_returns.csv")

    bab = pd.read_csv(bab_path, index_col=0, parse_dates=True)
    mkt = pd.read_csv(excess_path, index_col=0, parse_dates=True)

    if "BAB_Excess_Return" in bab.columns:
        bab_excess = bab["BAB_Excess_Return"]
    else:
        bab_excess = bab["BAB_Return"]

    mkt_excess = mkt["MSCI_World"]

    common = bab_excess.index.intersection(mkt_excess.index)
    bab_excess = bab_excess.loc[common]
    mkt_excess = mkt_excess.loc[common]

    logger.info(f"Aligned excess series: {len(common)} months (BAB {bab_excess.index.min()} to {bab_excess.index.max()})")
    return bab_excess, mkt_excess


def run_capm(y, x):
    """
    Simple OLS with intercept: y = a + b x + e
    """
    n = len(y)
    if n < 24:
        raise ValueError("Not enough observations for regression.")

    X = np.column_stack([np.ones(n), x.values])
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y.values)
    residuals = y.values - X @ beta_hat
    sigma2 = (residuals @ residuals) / (n - 2)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_stats = beta_hat / se_beta

    # Normal-approx p-values
    normal_cdf = np.vectorize(lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2))))
    p_vals = 2 * (1 - normal_cdf(np.abs(t_stats)))

    y_hat = X @ beta_hat
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    return {
        "alpha_monthly": beta_hat[0],
        "beta_mkt": beta_hat[1],
        "alpha_t": t_stats[0],
        "beta_t": t_stats[1],
        "alpha_p": p_vals[0],
        "beta_p": p_vals[1],
        "r2": r2,
        "n": n,
    }


def format_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main():
    # Ensure output directory exists BEFORE any file operations
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bab_excess, mkt_excess = load_excess_series()
    res = run_capm(bab_excess, mkt_excess)

    alpha_ann = res["alpha_monthly"] * 12
    table = pd.DataFrame(
        [
            {
                "Model": "CAPM (URTH)",
                "Alpha_Monthly": res["alpha_monthly"],
                "Alpha_Annualized": alpha_ann,
                "Alpha_t": res["alpha_t"],
                "Alpha_p": res["alpha_p"],
                "Beta_Mkt": res["beta_mkt"],
                "Beta_t": res["beta_t"],
                "Beta_p": res["beta_p"],
                "R2": res["r2"],
                "N": res["n"],
            }
        ]
    )

    out_csv = os.path.join(OUTPUT_DIR, "bab_factor_regressions.csv")
    table.to_csv(out_csv, index=False)
    logger.info(f"Saved regression diagnostics to {out_csv}")

    # Console summary
    print("\nCAPM Regression (BAB excess vs URTH excess)")
    print("===========================================")
    print(f"Observations: {res['n']}")
    print(f"Alpha (monthly): {res['alpha_monthly']*100:.3f}% {format_stars(res['alpha_p'])} (t={res['alpha_t']:.2f}, p={res['alpha_p']:.4f})")
    print(f"Alpha (annualized): {alpha_ann*100:.3f}%")
    print(f"Beta_mkt: {res['beta_mkt']:.3f} {format_stars(res['beta_p'])} (t={res['beta_t']:.2f}, p={res['beta_p']:.4f})")
    print(f"R^2: {res['r2']:.3f}")
    print("Note: A beta near 0 and insignificant is expected if BAB is properly market-neutral.")


if __name__ == "__main__":
    main()
