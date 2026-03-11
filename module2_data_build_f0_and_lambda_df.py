import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

from module1_data_standardize import standardize_data
from module2_factor_growth import init_growth_factor_value
from module2_factor_discount import init_discount_factor_value
from module2_factor_risk_premium import init_risk_premium_factor_value


'''
Important note on data leakage:
-------------------------------
No data leakage in this script. Any potential data leakage will be purely a function of what you feed it as inputs. This is solved for in econ_model.py
'''


def build_lambda_df_init(df_std, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS,
                      pca_growth, pca_discount, pca_risk_premium):

    all_cols = GROWTH_COLS + DISCOUNT_COLS + RISK_PREMIUM_COLS

    lambda_init = pd.DataFrame(
        index=all_cols,
        columns=['Growth', 'Discount', 'Risk_Premium'],
        dtype=float
    )

    proxies = {
        'Growth':       pca_growth,
        'Discount':     pca_discount,
        'Risk_Premium': pca_risk_premium,
    }

    # --- Correlate every series against each PCA time series ---
    # PCA is the best available proxy for each latent factor prior to full estimation
    # NaNs handled by aligning on common non-NaN dates per pair
    for series in all_cols:
        for factor, proxy in proxies.items():
            aligned = pd.concat([df_std[series], proxy], axis=1).dropna()
            if len(aligned) > 12: #REQUIRES AT LEAST 12 MONTHS OF DATA 
                lambda_init.loc[series, factor] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            else:
                # Insufficient overlapping observations — set to near zero
                lambda_init.loc[series, factor] = 0.0

    # --- Report ---
    logger.info('Lambda Init Matrix Complete')

    return lambda_init


def build_F0(growth_init, discount_init, risk_prem_init):

    # --- Assemble initial state vector ---
    # F0 = starting position of each latent factor at t=0
    # Used to seed the Kalman filter before estimation begins
    # [growth, discount, risk_premium]
    F0 = [growth_init, discount_init, risk_prem_init]
    logger.info('=' * 65)
    logger.info(f'F0 Initial State Vector')
    logger.info(f'  Growth:       {growth_init:.4f}')
    logger.info(f'  Discount:     {discount_init:.4f}')
    logger.info(f'  Risk Premium: {risk_prem_init:.4f}')
    logger.info(f'  F0:           {[round(v, 4) for v in F0]}')

    return F0





#
# RUNNING THE CODE:
#

# --- Load and standardize ---
#START_YEAR = 1970
#REGRESSION_TARGET = 'L5_sp500_tr_3m'
#FWD_months = 3
#r2_threshold      = 0.05 # R2-value threshold for factor includes
#pval_threshold    = 0.10 # P-value threshold for factor includes
#tiebreaker_gap    = 0.05 # tiebreaker gap for R2-val / P-val, if inside this gap, we default to manual, static factor bucket list
#df = pd.read_csv('all_econ_data.csv', index_col='date', parse_dates=True)
#df_std, TARGET_VARIABLE, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED_COLS = standardize_data(
    df,
    REGRESSION_TARGET = REGRESSION_TARGET,
    forward_months = FWD_months,
    r2_threshold   = r2_threshold,
    pval_threshold = pval_threshold,
    tiebreaker_gap = tiebreaker_gap,
#)

# --- Get Principal Component Analysis time series from each factor init ---
# These are the best composite proxies for each latent factor
# Far better than using a single series as proxy

#growth_init,    pca_growth       = init_growth_factor_value(df_std, GROWTH_COLS, verbose=False)
#discount_init,  pca_discount     = init_discount_factor_value(df_std, DISCOUNT_COLS, verbose=False)
#risk_prem_init, pca_risk_premium = init_risk_premium_factor_value(df_std, RISK_PREMIUM_COLS, verbose=False)

# --- Build FO ---
#F0_init = build_F0(growth_init, discount_init, risk_prem_init)
# --- Build Lambda df ---
#lambda_init = build_lambda_df_init(df_std, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, pca_growth, pca_discount, pca_risk_premium)
#lambda_init.to_csv('lambda_init.csv')
#print('\nLambda init matrix saved to lambda_init.csv')
