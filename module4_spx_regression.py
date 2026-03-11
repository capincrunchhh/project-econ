import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import logging
logger = logging.getLogger(__name__)


def run_spx_regression(factors, spx, forward_months):
    """
    Regress forward SPX total returns on DFM factor scores.

    Observation equation:
        spx_forward(t + forward_months) = β0
                                        + β1 × Growth(t)
                                        + β2 × Discount(t)
                                        + β3 × RiskPrem(t)
                                        + ε(t)

    Factor scores at time t predict SPX total return over the next forward_months.
    The forward shift is applied explicitly — sp500_tr series are trailing by
    construction, so shifting back aligns factors with the return they should predict.

    Observation strategy (automatic):
        Stride (default): non-overlapping every forward_months rows — clean independence,
                          no autocorrelation, but thin at long horizons
        HAC fallback:     all observations with Newey-West corrected SEs, used when
                          stride yields < MIN_STRIDE_OBS independent observations

    Dynamic factor columns — supports 3-factor baseline and any n-factor extension
    (e.g. 4-factor composite from Step 9a). All factor columns in the input
    dataframe are automatically detected and used.

    -------------------------------------------------------------------------
    Parameters
    ----------
    factors        : pd.DataFrame — factor scores (T x n) from EM algorithm, standardized
    spx            : pd.DataFrame — SPX return series, raw decimal (e.g. 0.15 = 15%)
    forward_months : int          — how many months forward to shift SPX return

    -------------------------------------------------------------------------
    Returns
    -------
    results : dict
        model         — fitted statsmodels OLS model object
        df_regression — aligned dataframe used for regression
        df_predict    — full factor score history with predicted returns
        summary       — regression summary table
        use_hac       — bool, True if HAC fallback fired
        obs_method    — str, description of observation strategy used
    """

    # --- Align factor scores and SPX returns to common date range ---
    df = factors.join(spx, how='left')

    # --- Dynamic factor columns — detects all columns in factors input ---
    # supports 3-factor baseline (Growth, Discount, Risk_Premium) and
    # any n-factor extension passed in from Step 9a composite testing
    factor_cols = list(factors.columns)

    # --- Shift SPX return forward by forward_months ---
    spx_col = spx.columns[0]
    df['spx_forward'] = df[spx_col].shift(-forward_months)

    # --- Drop rows where either factors or target are NaN ---
    df_reg = df.dropna(subset=factor_cols + ['spx_forward'])

    # --- Observation strategy: stride vs. HAC ---
    # Both methods address autocorrelation from overlapping return windows, just differently:
    #
    # Stride (default): eliminates autocorrelation by only using every forward_months-th row
    #                   so windows are fully non-overlapping and independent by construction
    #                   preferred when enough observations are available (>= MIN_STRIDE_OBS)
    #
    # HAC fallback:     keeps all observations but corrects standard errors via Newey-West
    #                   autocorrelation is still present in the data but accounted for in inference
    #                   fires automatically when stride would yield < MIN_STRIDE_OBS observations
    #                   (e.g. 12m horizon from 1970 gives ~55 stride obs — too thin)
    MIN_STRIDE_OBS = 80

    df_reg_stride = df_reg.iloc[::forward_months]

    if len(df_reg_stride) >= MIN_STRIDE_OBS:
        df_reg     = df_reg_stride
        use_hac    = False
        obs_method = f'stride ({forward_months}m non-overlapping)'
    else:
        use_hac    = True
        obs_method = f'HAC (Newey-West, all obs, lags={forward_months})'

    logger.info('')
    logger.info('Regression dataset:')
    logger.info(f'  Total months available:           {len(df)}')
    logger.info(f'  Months used in regression:        {len(df_reg)}')
    logger.info(f'  Observation method:               {obs_method}')
    logger.info(f'  Date range: {df_reg.index[0].strftime("%Y-%m")} to {df_reg.index[-1].strftime("%Y-%m")}')
    logger.info(f'  Target series:                    {spx_col}')
    logger.info(f'  Forward horizon:                  {forward_months} months')
    logger.info(f'  Factor columns ({len(factor_cols)}):             {factor_cols}')

    # --- Build OLS regression ---
    X = sm.add_constant(df_reg[factor_cols])
    y = df_reg['spx_forward']

    if use_hac:
        model = sm.OLS(y, X).fit(cov_type='HAC',
                                  cov_kwds={'maxlags': forward_months,
                                            'use_correction': True})
    else:
        model = sm.OLS(y, X).fit()

    # --- Print regression summary ---
    logger.info('=' * 65)
    logger.info('OLS Regression: Forward SPX Total Return on DFM Factors')
    logger.info('=' * 65)
    logger.debug('\n' + str(model.summary()))
    logger.debug('')

    # --- Print condensed key statistics ---
    logger.info('Key Statistics:')
    logger.info(f'  R²:              {model.rsquared:.4f}')
    logger.info(f'  Adjusted R²:     {model.rsquared_adj:.4f}')
    logger.info(f'  F-statistic:     {model.fvalue:.2f}  (p={model.f_pvalue:.4f})')
    logger.info(f'  Observations:    {int(model.nobs)}')
    logger.info(f'  Method:          {obs_method}')
    logger.debug('')
    logger.debug('Coefficients:')
    logger.debug(f'  {"Variable":<20} {"Beta":>8} {"Std Err":>8} {"t-stat":>8} {"p-value":>8} {"Significant":>12}')
    logger.debug('  ' + '-' * 70)
    for var in model.params.index:
        beta  = model.params[var]
        se    = model.bse[var]
        tstat = model.tvalues[var]
        pval  = model.pvalues[var]
        sig   = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        logger.debug(f'  {var:<20} {beta:>8.4f} {se:>8.4f} {tstat:>8.3f} {pval:>8.4f} {sig:>12}')
    logger.debug('')

    # --- Generate predictions on full factor history ---
    X_full  = sm.add_constant(factors[factor_cols])
    pred    = model.get_prediction(X_full)
    pred_df = pred.summary_frame(alpha=0.05)

    df_predict = factors.copy()
    df_predict['predicted_spx_return'] = pred_df['mean']
    df_predict['ci_lower_95']          = pred_df['obs_ci_lower']
    df_predict['ci_upper_95']          = pred_df['obs_ci_upper']
    df_predict['realized_spx_return']  = df['spx_forward']

    # --- Print current prediction ---
    latest_date  = factors.index[-1]
    latest_pred  = df_predict['predicted_spx_return'].iloc[-1]
    latest_lower = df_predict['ci_lower_95'].iloc[-1]
    latest_upper = df_predict['ci_upper_95'].iloc[-1]

    logger.debug('=' * 65)
    logger.debug(f'First-Pass Prediction (as of {latest_date.strftime("%Y-%m")}):')
    logger.debug(f'  Predicted {forward_months}-month SPX total return: {latest_pred:+.1%}')
    logger.debug(f'  95% prediction interval:             [{latest_lower:+.1%}, {latest_upper:+.1%}]')
    logger.debug('=' * 65)

    # --- Save outputs ---

    return {
        'model'        : model,
        'df_regression': df_reg,
        'df_predict'   : df_predict,
        'summary'      : model.summary(),
        'use_hac'      : use_hac,
        'obs_method'   : obs_method,
    }





#
# RUNNING THE CODE
#

# --- Load factor scores ---
# shape (T, n) — Growth, Discount, Risk_Premium (+ optional composites) at every month
# output of EM algorithm, already on standardized scale (mean=0, std=1)

#factors = pd.read_csv('factor_scores.csv', index_col='date', parse_dates=True)

# --- Load SPX total return from raw (unstandardized) data ---
# use raw not standardized so predictions are interpretable as actual return percentages
# swap column name to match desired return interval:
#   L5_sp500_tr_1m  — month-over-month,  pair with FWD_months=1
#   L5_sp500_tr_3m  — 3-month trailing,  pair with FWD_months=3
#   L5_sp500_tr_yoy — 12-month trailing, pair with FWD_months=12

#REGRESSION_TARGET = 'L5_sp500_tr_3m'
#FWD_months = 3
#spx = pd.read_csv('all_econ_data.csv', index_col='date', parse_dates=True)[[REGRESSION_TARGET]]
#results = run_spx_regression(
#    factors        = factors,
#    spx            = spx,
#    forward_months = FWD_months,
#)