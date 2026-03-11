import pandas as pd
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,                          # INFO in production, DEBUG when investigating
    format='%(message)s',
    handlers=[
        logging.FileHandler('econ_model.log'),   # always write to file
        logging.StreamHandler()                  # also print to terminal
    ]
)

from API_keys import EIA_api_key, FRED_api_key, BEA_api_key, BLS_api_key
from module0_data_get_all import DFM_master_data_get
from module1_data_standardize import standardize_data
from module2_factor_growth import init_growth_factor_value
from module2_factor_discount import init_discount_factor_value
from module2_factor_risk_premium import init_risk_premium_factor_value
from module2_data_build_f0_and_lambda_df import build_lambda_df_init, build_F0
from module3_EM_algo import run_em_dfm
from module3_walkforward_em import run_walkforward_em
from module4_spx_regression import run_spx_regression
from module5_fundamental_valuation import run_gordon_growth_valuation
from module6_walk_forward_optimization import run_kalman_regression
from module7_final_results import run_final_synthesis
from module8_added_factors import run_composite_factor_analysis
from module9_loop import run_factor_addition_loop


'''
Important note on data leakage:
-------------------------------
2 remaining potential sources of data leakage exist: 

1. Vintage/Revision Leakage
This is probably the most significant remaining issue. FRED serves the current revised vintage of every series by default. So when the model trains on, say, 
2008 GDP data, it's using the 2024-revised estimate — which incorporates subsequent benchmark revisions, methodological changes, and seasonal adjustment updates 
that were not available in 2008. A practitioner in 2008 would have seen a materially different number. The canonical solution is to use ALFRED (Archival FRED), 
which stores real-time vintages. This is genuinely hard to implement and most academic papers acknowledge this limitation rather than solving it. For these 
purposes, the directional signal is unlikely to be destroyed by revisions, but the exact coefficient magnitudes may be inflated.

2. Intra-Month Timing Leakage
Every series is normalized to the 1st of the month, but releases happen on specific days. In practice, on January 15th you'd have CPI (released ~Jan 15) but 
not payrolls (released ~Jan 3, so actually available) and not PCE (released ~Jan 31, so not yet available). Current monthly shift in module0 approximates this 
but doesn't distinguish between a series released on the 3rd vs. the 31st of the following month. For a 1-month horizon model this could matter at the margin, 
though the effect is likely small relative to the monthly aggregation.
'''

# Potentail SPX regression targets
#'L5_sp500_tr_1m'
#'L5_sp500_tr_3m'
#'L5_sp500_tr_6m'
#'L5_sp500_tr_yoy'

# Parameters
#------------
START_YEAR = 1970 # data prior to 1970 gets patchier
REGRESSION_TARGET = 'L5_sp500_tr_6m'
FWD_months = 1 # how many months are we trying to predict? 1m = predict next 1m return
r2_threshold      = 0.05 # R2-value threshold for inclusion in original 3 factors
pval_threshold    = 0.05 # P-value threshold for inclusion in original 3 factors
tiebreaker_gap    = 0.05 # tiebreaker gap for R2-val / P-val, if inside this gap, we default to manual, static factor bucket list
oos_new_factor_addition_threshold = .10 # adds in new factor only if the re-run of EM algo + walk-fwd Kalman regression adds > .10 OSS R2
oos_start_year = 1990 # sets start year for the out of sample window. this impacts the standardized data walk forward dataframe and 
                        #walk-forward regression in the EM algo. later start year = better starting coefficients = higher R2 but less OOS predictions
n_iter = 200 # number of iterations on EM algo (affects runtime)
em_tol = 1e-4 # tolerances for EM algo (affects runtime). tol=1e-4 means stop when change is less than 0.0001, which is still extremely tight 
            #relative to a log-likelihood of -62000. The factor scores at 1e-4 vs 1e-6 convergence will be numerically identical for practical purposes.

# Pre-callibrated configurations:
#--------------------------------

# TARGET=12m SPX | FWD=12 | "Macro Cycle"
# Long-horizon factors and prediction. likely that HAC autocorrelation fix correction fires (~55 obs, below stride threshold).
# Designed to include slow-moving series in factors that fail the 6m filter: profit cycles, capex, fiscal, wealth effects.
# Use for cycle-peak/trough identification and long-duration positioning.

# TARGET=6m SPX | FWD=6 | "Strategic / Regime Monitor"
# Factors selected and evaluated on the same 6m horizon. Clean, non-overlapping windows (111 obs). 
# Use for portfolio positioning and regime classification. Most reliable valuation read.

# TARGET=6m SPX | FWD=3 | "Intermediate"
# 6m-quality factors applied to a 3m prediction horizon. Bridge between regime and tactical signals. 
# Strong corrected OOS R² (0.46) for a 3m macro model. Predictions use built in bias correction (1.35x too spread).

# TARGET=6m SPX | FWD=1 | "Tactical"
# 6m-quality factors applied to monthly predictions. Highest OOS R² of the three (0.65+), 671 obs, fastest Kalman beta adaptation.
# Use for monthly positioning and early regime shift detection.


# Step 0 - import data
df = DFM_master_data_get(START_YEAR)
df.to_csv('all_econ_data.csv')
script_start = time.time()
# Step 1 - standardize data
(   df_std, df_std_wf, TARGET_VARIABLE,
    GROWTH_COLS,    DISCOUNT_COLS,    RISK_PREMIUM_COLS,    CURRENTLY_UNUSED_COLS,    df_ranked,
    GROWTH_COLS_WF, DISCOUNT_COLS_WF, RISK_PREMIUM_COLS_WF, CURRENTLY_UNUSED_COLS_WF, df_ranked_wf,) = standardize_data(
    df                = df,
    REGRESSION_TARGET = REGRESSION_TARGET,
    forward_months    = FWD_months,
    r2_threshold      = r2_threshold,
    pval_threshold    = pval_threshold,
    tiebreaker_gap    = tiebreaker_gap,
    oos_start_year    = oos_start_year,
)
df_std.to_csv('all_econ_data_standardized.csv')
df_std_wf.to_csv('all_econ_data_standardized_wf.csv')
# Step 2 - create F0 and Lambda matrix 
# (full sample, for Steps 3-5)
growth_init,    pca_growth       = init_growth_factor_value(df_std, GROWTH_COLS)
discount_init,  pca_discount     = init_discount_factor_value(df_std, DISCOUNT_COLS)
risk_prem_init, pca_risk_premium = init_risk_premium_factor_value(df_std, RISK_PREMIUM_COLS)
F0         = build_F0(growth_init, discount_init, risk_prem_init)
lambda_df  = build_lambda_df_init(df_std, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, pca_growth, pca_discount, pca_risk_premium)
lambda_df.to_csv('lambda_init.csv')
# (data leakage free, for Steps 6+)
growth_init_wf,    pca_growth_wf       = init_growth_factor_value(df_std_wf, GROWTH_COLS_WF)
discount_init_wf,  pca_discount_wf     = init_discount_factor_value(df_std_wf, DISCOUNT_COLS_WF)
risk_prem_init_wf, pca_risk_premium_wf = init_risk_premium_factor_value(df_std_wf, RISK_PREMIUM_COLS_WF)
F0_wf        = build_F0(growth_init_wf, discount_init_wf, risk_prem_init_wf)
lambda_df_wf = build_lambda_df_init(df_std_wf, GROWTH_COLS_WF, DISCOUNT_COLS_WF, RISK_PREMIUM_COLS_WF, pca_growth_wf, pca_discount_wf, pca_risk_premium_wf)
lambda_df_wf.to_csv('lambda_init_wf.csv')
# Step 3 - EM algo (full series)
all_cols = GROWTH_COLS + DISCOUNT_COLS + RISK_PREMIUM_COLS
Y        = df_std[all_cols].values.astype(float)
logger.debug(f'  Data shape:   {Y.shape}')
logger.debug(f'  F0:           {[round(v, 3) for v in F0]}')
logger.debug(f'  Lambda shape: {lambda_df.values.shape}')
results = run_em_dfm(
    Y           = Y,
    lambda_init = lambda_df.values,
    F0          = F0,
    n_iter      = n_iter,
    tol         = em_tol,
)
dates    = df_std.index
F_smooth = pd.DataFrame(results['F_smooth'], index=dates, columns=['Growth', 'Discount', 'Risk_Premium'])
Lambda   = pd.DataFrame(results['Lambda'],   index=all_cols, columns=['Growth', 'Discount', 'Risk_Premium'])
F_smooth.to_csv('factor_scores.csv')
Lambda.to_csv('lambda_estimated.csv')
logger.info(f'Final log-likelihood: {results["ll_history"][-1]:.2f}')
# Step 4 - SPX Regression (full series)
step4_results = run_spx_regression(
    factors        = F_smooth,
    spx            = df[[REGRESSION_TARGET]],
    forward_months = FWD_months
)
# Step 5 - SPX Valuation (full series)
valuation_results = run_gordon_growth_valuation(
    factors            = F_smooth,
    raw                = df,
    mapping_start_year = 1990, # for this function 1970's stagflation and volcker regime adds instability to valuation
    smoothing_months   = 12,
    enable_graph       = False,
)
# Step 6 - Walk-Fwd Kalman Regression (Time-Varying Parameters) - Uses walk-fwd EM algo which is data-leakage free
F_smooth_wf, final_G, final_D, final_RP = run_walkforward_em(
    df_std         = df_std_wf,
    df_std_full    = df_std,
    all_cols       = GROWTH_COLS_WF + DISCOUNT_COLS_WF + RISK_PREMIUM_COLS_WF,
    lambda_init    = lambda_df_wf.values,
    F0_init        = F0_wf,
    oos_start_year = oos_start_year,
    n_iter          = n_iter,
    tol             = em_tol,
    df_raw         = df,
    target_col     = REGRESSION_TARGET,
    r2_threshold   = r2_threshold,
    pval_threshold = pval_threshold,
    tiebreaker_gap = tiebreaker_gap,
    forward_months = FWD_months,
    growth_cols    = GROWTH_COLS_WF,        # seed for current_G tracker
    discount_cols  = DISCOUNT_COLS_WF,      # seed for current_D tracker
    risk_prem_cols = RISK_PREMIUM_COLS_WF,  # seed for current_RP tracker
)
# Update bucket lists to reflect final walk-forward state — promoted series drop off unused as they get allocated to factors
final_wf_cols            = final_G + final_D + final_RP
CURRENTLY_UNUSED_COLS_WF = [c for c in CURRENTLY_UNUSED_COLS_WF if c not in final_wf_cols]
FACTOR_COLS_WF           = [final_G, final_D, final_RP]
kalman_results = run_kalman_regression(
    F_smooth          = F_smooth_wf,
    df_raw            = df,
    REGRESSION_TARGET = REGRESSION_TARGET,
    forward_months    = FWD_months,
    start_year        = START_YEAR,
    beta_drift_q      = 0.001,
    obs_noise_r       = None,
    in_sample_r2      = step4_results['model'].rsquared,
    enable_graph      = False,
)
# Step 7 - Final Results
synthesis = run_final_synthesis(
    step4_results     = step4_results,
    step5_results     = valuation_results,
    step6_results     = kalman_results,
    forward_months    = FWD_months,
    regression_target = REGRESSION_TARGET,
    F_smooth          = F_smooth_wf,
    Lambda            = Lambda,
    df_raw            = df,
    df_ranked         = df_ranked_wf,
)
# Step 8 - Recycle unused data into additional factors
step8_results = run_composite_factor_analysis(
    df_std                = df_std_wf,
    df_raw                = df,
    CURRENTLY_UNUSED_COLS = CURRENTLY_UNUSED_COLS_WF,
    REGRESSION_TARGET     = REGRESSION_TARGET,
    forward_months        = FWD_months,
    F_smooth              = F_smooth_wf,
    oos_start_year        = oos_start_year,
)
# Step 9 — Iterative new factor addition
step9_results = run_factor_addition_loop(
    step8_results             = step8_results,
    F_smooth                  = F_smooth_wf,
    Lambda                    = Lambda,
    step4_results             = step4_results,
    kalman_results            = kalman_results,
    valuation_results         = valuation_results,
    df_raw                    = df,
    df_std                    = df_std_wf,
    df_std_full               = df_std, 
    df_ranked                 = df_ranked_wf,
    CURRENTLY_UNUSED_COLS     = CURRENTLY_UNUSED_COLS_WF,
    FACTOR_COLS               = FACTOR_COLS_WF,
    FACTOR_NAMES              = ['Growth', 'Discount', 'Risk_Premium'], # seed values for first loop
    pca_proxies               = {'Growth': pca_growth_wf, 'Discount': pca_discount_wf, 'Risk_Premium': pca_risk_premium_wf},
    REGRESSION_TARGET         = REGRESSION_TARGET,
    forward_months            = FWD_months,
    start_year                = START_YEAR,
    factor_addition_threshold = oos_new_factor_addition_threshold,
    oos_start_year = oos_start_year
)
# Step 10 — Final synthesis with best model
final_F_smooth       = step9_results['F_smooth']
final_Lambda         = step9_results['Lambda']
final_step4_results  = step9_results['step4_results']
final_kalman_results = step9_results['kalman_results']
final_factor_names   = step9_results['factor_names']
final_synthesis = run_final_synthesis(
    step4_results     = final_step4_results,    
    step5_results     = valuation_results,
    step6_results     = final_kalman_results,
    forward_months    = FWD_months,
    regression_target = REGRESSION_TARGET,
    F_smooth          = final_F_smooth, 
    Lambda            = final_Lambda,
    df_raw            = df,
    df_ranked         = df_ranked_wf,
)
final_F_smooth.to_csv('factor_scores_final.csv')
final_Lambda.to_csv('lambda_estimated_final.csv')
final_kalman_results['df_predictions'].to_csv('predictions_final.csv')
script_end = time.time()
logger.info(f'Total runtime:  {(script_end - script_start) / 60:.1f} min')