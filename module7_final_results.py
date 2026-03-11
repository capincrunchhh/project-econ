import pandas as pd
import numpy as np
import yfinance as yf
import logging
logger = logging.getLogger(__name__)


def get_live_spx():
    """Pull live SPX price from Yahoo Finance."""
    ticker = yf.Ticker('^GSPC')
    return ticker.fast_info['last_price']


def run_final_synthesis(
        step4_results,
        step5_results,
        step6_results,
        forward_months,
        regression_target,
        F_smooth,
        Lambda,
        df_raw,
        df_ranked,
):
    """
    Step 7 — Final Synthesis Report

    Synthesizes outputs from Steps 3, 4, 5, and 6 into a single
    plain-english investment summary. No new computation —
    purely a reporting and interpretation layer.

    Pulls live SPX price from Yahoo Finance to compute realized
    return since prediction was made and remaining implied upside.

    Parameters
    ----------
    step4_results      : dict       — return value from run_spx_regression()
    step5_results      : dict       — return value from run_gordon_growth_valuation()
    step6_results      : dict       — return value from run_kalman_regression()
    forward_months     : int        — prediction horizon in months (dynamic)
    regression_target  : str        — e.g. 'L5_sp500_tr_3m'
    F_smooth           : DataFrame  — Kalman-smoothed factor scores from EM algo
    Lambda             : DataFrame  — estimated loading matrix from EM algo
    df_raw             : DataFrame  — raw (unstandardized) econ data
    """

    # =========================================================
    # EXTRACT INPUTS
    # =========================================================

    # Step 4
    model          = step4_results['model']
    s4_r2          = model.rsquared
    s4_r2_adj      = model.rsquared_adj
    s4_fstat       = model.fvalue
    s4_fpval       = model.f_pvalue
    s4_nobs        = int(model.nobs)
    s4_params      = model.params
    s4_pvalues     = model.pvalues
    s4_pred_df     = step4_results['df_predict']
    s4_current     = s4_pred_df['predicted_spx_return'].iloc[-1]
    s4_ci_lower    = s4_pred_df['ci_lower_95'].iloc[-1]
    s4_ci_upper    = s4_pred_df['ci_upper_95'].iloc[-1]

    # Step 5
    s5_fair        = step5_results['fair_value']
    s5_actual      = step5_results['current_spx']
    s5_pct         = step5_results['over_under_pct']
    s5_pctile      = step5_results['over_under_pctile']
    s5_zscore      = step5_results['over_under_zscore']

    # Step 6
    s6_oos_r2      = step6_results['oos_r2']
    s6_oos_r2_bc   = step6_results['oos_r2_bc']
    s6_dir_acc     = step6_results['directional_acc']
    s6_dir_acc_bc  = step6_results['dir_acc_bc']
    s6_mae         = step6_results['mae']
    s6_mae_bc      = step6_results['mae_bc']
    s6_mono        = step6_results['monotonicity']
    s6_mean_bias   = step6_results['mean_bias']
    s6_scale       = step6_results['scale_factor']
    s6_pred_raw    = step6_results['current_pred']
    s6_pred_bc     = step6_results['current_pred_bc']
    s6_betas       = step6_results['current_betas']
    s6_quintiles   = step6_results['quintile_stats']
    s6_y_mean      = step6_results['y_mean']

    # Derive eval start year from predictions dataframe
    eval_start_year = step6_results['df_predictions'].dropna(
        subset=['realized_return']).index[0].year

    # =========================================================
    # STEP 3 — FACTOR DECOMPOSITION PREP
    # =========================================================
    # Lambda maps factor space → data space (estimated by EM)
    # F_smooth are Kalman-smoothed factor scores
    # Standardize raw data using each indicator's own history
    # to get current deviation from mean in std dev units.
    # Use last non-null per indicator to handle data lag.

    indicator_cols = [c for c in Lambda.index if c in df_raw.columns]
    df_sub         = df_raw[indicator_cols].copy()

    df_std_decomp  = pd.DataFrame(index=df_sub.index,
                                  columns=df_sub.columns, dtype=float)
    for col in df_sub.columns:
        series = df_sub[col].dropna()
        mu     = series.mean()
        sigma  = series.std()
        if sigma > 0:
            df_std_decomp[col] = (df_sub[col] - mu) / sigma
        else:
            df_std_decomp[col] = 0.0

    current_std = df_std_decomp.apply(lambda col: col.dropna().iloc[-1]
                                      if len(col.dropna()) > 0 else np.nan)
    last_dates  = df_std_decomp.apply(lambda col: col.dropna().index[-1]
                                      if len(col.dropna()) > 0 else pd.NaT)

    # =========================================================
    # PREDICTION WINDOW
    # =========================================================

    pred_start_date = F_smooth.index[-1]
    pred_end_date     = pred_start_date + pd.DateOffset(months=forward_months)
    spx_at_pred_start = s5_actual

    try:
        spx_live      = get_live_spx()
        spx_live_date = pd.Timestamp.today().strftime('%b %d %Y')
        live_ok       = True
    except Exception:
        spx_live      = spx_at_pred_start
        spx_live_date = pred_start_date.strftime('%b %Y')
        live_ok       = False

    realized_so_far = (spx_live / spx_at_pred_start) - 1
    days_remaining  = max((pred_end_date - pd.Timestamp.today()).days, 0)

    # =========================================================
    # BUILD QUINTILE THRESHOLDS FROM PREDICTION HISTORY
    # =========================================================

    df_eval   = step6_results['df_predictions'].dropna(
                    subset=['realized_return', 'predicted_return'])

    pred_vals  = sorted(df_eval['predicted_return'].values)
    n          = len(pred_vals)
    q_size     = n // 5
    thresholds = [pred_vals[q_size], pred_vals[2*q_size],
                  pred_vals[3*q_size], pred_vals[4*q_size]]

    def assign_q(p):
        if p <= thresholds[0]:   return 1
        elif p <= thresholds[1]: return 2
        elif p <= thresholds[2]: return 3
        elif p <= thresholds[3]: return 4
        else:                    return 5

    def quintile_label(pred):
        q = assign_q(pred)
        labels = {1: 'Quintile 1 — MOST BEARISH', 2: 'Quintile 2 — MILDLY BEARISH',
                  3: 'Quintile 3 — NEUTRAL',      4: 'Quintile 4 — MILDLY BULLISH',
                  5: 'Quintile 5 — MOST BULLISH'}
        return labels[q], q

    current_quintile_label, current_q = quintile_label(s6_pred_raw)

    # =========================================================
    # UPSIDE / DOWNSIDE MAGNITUDES FROM CURRENT QUINTILE
    # =========================================================

    df_eval      = df_eval.copy()
    df_eval['q'] = df_eval['predicted_return'].apply(assign_q)
    q_months     = df_eval[df_eval['q'] == current_q]['realized_return']
    pos_months   = q_months[q_months >= 0]
    neg_months   = q_months[q_months <  0]

    upside_mag   = pos_months.mean() if len(pos_months) > 0 else np.nan
    downside_mag = neg_months.mean() if len(neg_months) > 0 else np.nan

    # =========================================================
    # QUINTILE HISTORY STATS
    # =========================================================

    if s6_quintiles is not None:
        q_mean_realized = s6_quintiles['mean_realized'].iloc[current_q - 1]
        q_pct_positive  = s6_quintiles['pct_positive'].iloc[current_q - 1]
        q_count         = int(s6_quintiles['count'].iloc[current_q - 1])
    else:
        q_mean_realized = np.nan
        q_pct_positive  = np.nan
        q_count         = 0

    # =========================================================
    # SCENARIO PROBABILITIES AND CENTRAL CASES
    # =========================================================

    upside_prob     = q_pct_positive
    downside_prob   = 1 - q_pct_positive
    central_case_up = (s6_pred_bc + q_mean_realized) / 2

    if s6_pred_bc < 0:
        central_case_down = (s6_pred_bc + downside_mag) / 2
    else:
        central_case_down = downside_mag

    remaining_upside = central_case_up   - realized_so_far
    remaining_down   = central_case_down - realized_so_far

    # =========================================================
    # HELPER: signal strength labels
    # =========================================================

    def r2_label(r2):
        if r2 >= 0.20: return 'STRONG'
        if r2 >= 0.10: return 'MODERATE'
        if r2 >= 0.05: return 'WEAK'
        return 'VERY WEAK'

    def pval_label(p):
        if p < 0.01:  return '*** (p<0.01) highly significant'
        if p < 0.05:  return '**  (p<0.05) significant'
        if p < 0.10:  return '*   (p<0.10) marginally significant'
        return f'    (p={p:.3f}) NOT significant'

    def oos_r2_label(r2):
        if r2 >= 0.10:  return 'PASS — beats naive benchmark'
        if r2 >= 0.00:  return 'MARGINAL — roughly ties naive benchmark'
        if r2 >= -0.10: return 'SOFT FAIL — slightly below naive benchmark'
        return 'FAIL — worse than naive benchmark'

    def dir_acc_label(acc):
        if acc >= 0.75: return 'EXCELLENT'
        if acc >= 0.65: return 'GOOD'
        if acc >= 0.55: return 'USEFUL'
        return 'POOR — near random'

    def zscore_label(z):
        if z >= 2.0:    return 'SIGNIFICANTLY OVERVALUED'
        if z >= 1.0:    return 'MODERATELY OVERVALUED'
        if z >= 0.0:    return 'SLIGHTLY OVERVALUED'
        if z >= -1.0:   return 'SLIGHTLY UNDERVALUED'
        if z >= -2.0:   return 'MODERATELY UNDERVALUED'
        return 'SIGNIFICANTLY UNDERVALUED'

    # =========================================================
    # QUADRANT CLASSIFICATION
    # =========================================================

    macro_bullish = current_q >= 3
    val_cheap     = s5_zscore < 1.0

    if macro_bullish and val_cheap:
        quadrant_val   = 'CHEAP VALUATION'
        quadrant_macro = 'MACRO BULLISH'
        quadrant_label = 'Strong conviction buy'
        quadrant_desc  = 'Best risk/reward'
    elif macro_bullish and not val_cheap:
        quadrant_val   = 'EXPENSIVE VALUATION'
        quadrant_macro = 'MACRO BULLISH'
        quadrant_label = 'Momentum buy, vol risk'
        quadrant_desc  = 'Returns likely positive but ceiling is lower'
    elif not macro_bullish and val_cheap:
        quadrant_val   = 'CHEAP VALUATION'
        quadrant_macro = 'MACRO BEARISH'
        quadrant_label = 'Potential value trap'
        quadrant_desc  = 'Watch for turn'
    else:
        quadrant_val   = 'EXPENSIVE VALUATION'
        quadrant_macro = 'MACRO BEARISH'
        quadrant_label = 'Strong conviction sell'
        quadrant_desc  = 'Worst risk/reward'

    # =========================================================
    # CONSENSUS AND VALUATION OVERLAY
    # =========================================================

    signals = [s4_current, s6_pred_raw, s6_pred_bc]
    n_bull  = sum(1 for s in signals if s > 0)
    n_bear  = sum(1 for s in signals if s < 0)
    if n_bull == 3:
        consensus = 'BULLISH — all three signals agree positive'
    elif n_bear == 3:
        consensus = 'BEARISH — all three signals agree negative'
    elif n_bull == 2:
        consensus = 'LEANING BULLISH — 2 of 3 signals positive'
    else:
        consensus = 'LEANING BEARISH — 2 of 3 signals negative'

    if s5_zscore >= 1.5:
        val_overlay = f'NOTE: Valuation at {s5_zscore:+.2f}σ is a meaningful headwind'
    elif s5_zscore <= -1.5:
        val_overlay = f'NOTE: Valuation at {s5_zscore:+.2f}σ is a meaningful tailwind'
    else:
        val_overlay = f'NOTE: Valuation at {s5_zscore:+.2f}σ is roughly neutral'

    # =========================================================
    # PRINT REPORT
    # =========================================================

    sep = '=' * 65
    logger.info(sep)
    logger.info('FINAL SYNTHESIS REPORT')
    logger.info(f'Prediction horizon: {forward_months} months  |  Target: {regression_target}')
    logger.info(sep)

    # ----------------------------------------------------------
    # STEP 3 BLOCK — EM FACTOR DECOMPOSITION
    # ----------------------------------------------------------
    logger.debug('=' * 82)
    logger.debug('EM FACTOR DECOMPOSITION (Current Factor Drivers)')
    logger.debug('=' * 82)

    # --- Build factor membership lists from df_ranked ---
    factor_members = {'Growth': [], 'Discount': [], 'Risk_Premium': [], 'Unused': []}

    for _, row in df_ranked.iterrows():
        bucket = row['assigned_bucket']
        key    = 'Unused' if bucket == 'UNUSED' else bucket
        factor_members[key].append(row['series'])

    # The 3 anchor series are hardcoded into buckets in standardize_data and
    # excluded from df_ranked's test_cols — insert them at the front
    anchor_map = {
        'L4_gdp_yoy'          : 'Growth',
        'L0_treasury_10y'     : 'Discount',
        'L0_hy_credit_spread' : 'Risk_Premium',
    }
    all_ranked_series = set(df_ranked['series'].tolist())
    for series, key in anchor_map.items():
        if series not in all_ranked_series:
            factor_members[key].insert(0, series)

    em_series = set(Lambda.index)

    logger.debug('  Factor Membership')
    logger.debug(f'  {"─" * 78}')
    for label, key in [('Growth Factor',              'Growth'),
                        ('Discount Factor',            'Discount'),
                        ('Equity Risk Premium Factor', 'Risk_Premium'),
                        ('Unused',                     'Unused')]:
        members = factor_members[key]
        logger.debug(f'  {label} ({len(members)}):')
        for m in members:
            in_model = '  [in model]' if m in em_series else ''
            logger.debug(f'      {m}{in_model}')
        logger.debug('')

    logger.debug('')
    logger.debug('  Hist Weight:         loading estimated by EM algorithm (Lambda matrix)')
    logger.debug('                       direction and magnitude of how this indicator')
    logger.debug('                       historically co-moves with the factor')
    logger.debug('                       does not change run to run — fixed after EM convergence')
    logger.debug('')
    logger.debug('  Current (σ vs hist): where this indicator sits RIGHT NOW vs its own history')
    logger.debug('                       0 = at historical mean')
    logger.debug('                       +1 = one std dev above mean  (elevated)')
    logger.debug('                       -1 = one std dev below mean  (depressed)')
    logger.debug('                       ±2 = extreme historical reading')
    logger.debug('')
    logger.debug('  Current Weight:      Hist Weight × Current (σ vs hist)')
    logger.debug('                       directional influence of this indicator on the factor today')
    logger.debug('                       positive = pulling factor score higher')
    logger.debug('                       negative = pulling factor score lower')
    logger.debug('                       ranked by absolute magnitude — biggest movers shown first')
    logger.debug('')
    logger.debug('  NOTE: Current Weights are directional influence rankings, not a')
    logger.debug('  reconstruction of the factor score. Factor scores are produced by the')
    logger.debug('  Kalman smoother using the full time series path — not a simple weighted')
    logger.debug('  sum of current indicator readings. Sum of Current Weights will not')
    logger.debug('  equal the Kalman-smoothed factor score shown at the top of each block.')
    logger.debug('')
    logger.debug(f'  {"─" * 78}')
    logger.debug(f'  Factor scores as of {F_smooth.index[-1].strftime("%b %Y")}')
    logger.debug(f'  Indicator values as of each series most recent available date')
    logger.debug(f'  {"─" * 78}')

    for factor in ['Growth', 'Discount', 'Risk_Premium']:

        factor_score = F_smooth.iloc[-1][factor]
        loadings     = Lambda[factor]

        contributions = {}
        for indicator in loadings.index:
            if indicator in current_std.index:
                lam = loadings[indicator]
                val = current_std[indicator]
                if not np.isnan(lam) and not pd.isna(val):
                    contributions[indicator] = lam * val

        ranked = sorted(contributions.items(),
                        key=lambda x: abs(x[1]), reverse=True)

        logger.debug('')
        logger.debug(f'  {"─" * 78}')
        logger.debug(f'  {factor} Factor  |  Kalman-smoothed score: {factor_score:>+.4f}')
        logger.debug(f'  Top 5 contributors by absolute current weight  '
              f'({len(contributions)} total indicators in factor)')
        logger.debug(f'  {"─" * 78}')
        logger.debug(f'  {"Indicator":<35} {"Hist Weight":>11}  {"Curr (σ vs hist)":>16}  '
              f'{"Curr Weight":>12}  {"As of":>8}')
        logger.debug(f'  {"─" * 78}')

        for indicator, contrib in ranked[:5]:
            lam         = loadings[indicator]
            val         = current_std[indicator]
            as_of       = last_dates[indicator].strftime('%b %Y')
            direction   = '↑' if contrib > 0 else '↓'
            contrib_str = f'{contrib:>+11.4f} {direction}'
            logger.debug(f'  {indicator:<35} {lam:>+11.4f}  {val:>+16.4f}  '
                  f'{contrib_str:<14}  {as_of:>8}')

        total_contrib = sum(contributions.values())
        top5_contrib  = sum(c for _, c in ranked[:5])
        residual      = total_contrib - top5_contrib

        logger.debug(f'  {"─" * 78}')
        logger.debug(f'  {"(remaining indicators)":<35} {"":>11}  {"":>16}  {residual:>+11.4f}')
        logger.debug(f'  {"Sum of curr weights (approx)":<35} {"":>11}  {"":>16}  {total_contrib:>+11.4f}')
        logger.debug(f'  {"Kalman-smoothed factor score":<35} {"":>11}  {"":>16}  {factor_score:>+11.4f}')

    # ----------------------------------------------------------
    # STEP 4 BLOCK
    # ----------------------------------------------------------
    logger.info('=' * 65)
    logger.info('OLS REGRESSION (Statistical Validity)')
    logger.info('=' * 65)
    logger.info(f'  In-sample R²:        {s4_r2:.4f}  [{r2_label(s4_r2)}]')
    logger.info(f'  Adjusted R²:         {s4_r2_adj:.4f}')
    logger.info(f'  F-statistic:         {s4_fstat:.2f}  (p={s4_fpval:.4f})')
    logger.info(f'  Observations:        {s4_nobs}  (non-overlapping)')
    logger.info('')
    logger.info('  Factor significance:')
    for var in ['Growth', 'Discount', 'Risk_Premium']:
        beta = s4_params.get(var, np.nan)
        pval = s4_pvalues.get(var, np.nan)
        direction = 'bullish signal ↑' if beta < 0 else 'bearish signal ↑'
        logger.info(f'    {var:<16}  β={beta:+.4f}  {pval_label(pval)}  ({direction})')
    logger.info('')

    sig_count = sum(1 for v in ['Growth', 'Discount', 'Risk_Premium']
                    if s4_pvalues.get(v, 1) < 0.10)
    if s4_r2 >= 0.10 and sig_count >= 2 and s4_fpval < 0.05:
        s4_verdict = 'RELIABLE — R² acceptable, factors significant, model valid'
    elif s4_r2 >= 0.05 and sig_count >= 1:
        s4_verdict = 'PARTIALLY RELIABLE — weak R² but some factors significant'
    else:
        s4_verdict = 'UNRELIABLE — low R² and/or insignificant factors'

    logger.info(f'  ▶ Verdict:  {s4_verdict}')
    logger.info('')
    logger.info(f'  Step 4 current {forward_months}m prediction:  {s4_current:+.1%}')
    logger.info(f'  95% interval:  [{s4_ci_lower:+.1%},  {s4_ci_upper:+.1%}]')

    # ----------------------------------------------------------
    # STEP 5 BLOCK
    # ----------------------------------------------------------
    logger.info('=' * 65)
    logger.info('GORDON GROWTH VALUATION (Market Stretch)')
    logger.info('=' * 65)
    logger.info(f'  SPX Actual:          ${s5_actual:>8,.0f}')
    logger.info(f'  SPX Fair Value:      ${s5_fair:>8,.0f}')
    logger.info(f'  Over/Under (raw):    {s5_pct:>+.1%}')
    logger.info(f'  Historical pctile:   {s5_pctile:.0%}')
    logger.info(f'  Z-Score (primary):   {s5_zscore:>+.2f}σ  [{zscore_label(s5_zscore)}]')
    logger.info('')

    if s5_zscore >= 2.0:
        s5_implication = 'Historically, returns from this valuation level are poor'
    elif s5_zscore >= 1.0:
        s5_implication = 'Valuation is a headwind — returns tend to be below average'
    elif s5_zscore >= -1.0:
        s5_implication = 'Valuation is roughly neutral — not a strong headwind or tailwind'
    elif s5_zscore >= -2.0:
        s5_implication = 'Valuation is a tailwind — returns tend to be above average'
    else:
        s5_implication = 'Historically, returns from this valuation level are strong'

    logger.info(f'  ▶ Implication:  {s5_implication}')

    # ----------------------------------------------------------
    # STEP 6 BLOCK
    # ----------------------------------------------------------
    logger.info('=' * 65)
    logger.info('KALMAN REGRESSION (OOS Predictive Validity)')
    logger.info('=' * 65)
    logger.info('  Out-of-sample performance (raw predictions):')
    logger.info(f'    OOS R²:              {s6_oos_r2:>+.4f}  [{oos_r2_label(s6_oos_r2)}]')
    logger.info(f'    Directional acc:     {s6_dir_acc:.1%}  [{dir_acc_label(s6_dir_acc)}]')
    logger.info(f'    Mean Abs Error:                 {s6_mae:.1%}')
    logger.info('')
    logger.info('  Quintile ranking (definitive signal quality test):')
    logger.info(f'    Monotonicity:        {s6_mono:.0%}  '
          f'({"PERFECT — signal is rankable across all regimes" if s6_mono == 1.0 else "IMPERFECT — some quintile ordering breaks down"})')
    if s6_quintiles is not None:
        logger.info(f'    {"Quintile":<22} {"Mean Realized":>13} {"% Positive":>11}')
        logger.info('    ' + '-' * 48)
        for q, row in s6_quintiles.iterrows():
            marker = ' ◀ current' if (s6_quintiles.index.get_loc(q) == current_q - 1) else ''
            logger.info(f'    {str(q):<22} {row["mean_realized"]:>+12.1%} '
                  f'{row["pct_positive"]:>10.1%}{marker}')
    logger.info('')
    logger.info('  Bias analysis:')
    bias_desc = 'overpredicts' if s6_mean_bias > 0 else 'underpredicts'
    logger.info(f'    Mean error:          {s6_mean_bias:>+.4f}  (model {bias_desc} on average)')
    logger.info(f'    Scale factor:        {s6_scale:.2f}x  '
          f'({"predictions too compressed" if s6_scale > 1 else f"predictions {1/s6_scale:.1f}x too spread"})')
    logger.info('')
    logger.info('  After bias correction:')
    logger.info(f'    OOS R² (corrected):  {s6_oos_r2_bc:>+.4f}  [{oos_r2_label(s6_oos_r2_bc)}]')
    logger.info(f'    Directional acc:     {s6_dir_acc_bc:.1%}')
    logger.info(f'    MAE (corrected):     {s6_mae_bc:.1%}')
    logger.info('')
    logger.info('  Current factor betas:')
    for k, v in s6_betas.items():
        logger.info(f'    {k:<16}  β={v:+.4f}')
    logger.info('')
    logger.info(f'  ▶ Current quintile:    {current_quintile_label}')
    logger.info(f'     Avg realized return at this quintile:  '
          f'{q_mean_realized:+.1%}  '
          f'(since {eval_start_year}, n={q_count} months)')
    logger.info(f'     % of months positive at this quintile: '
          f'{q_pct_positive:.1%}')

    # ----------------------------------------------------------
    # PREDICTION SUMMARY
    # ----------------------------------------------------------
    logger.debug('')
    logger.debug('━' * 65)
    logger.debug(f'PREDICTION SUMMARY — {forward_months}-MONTH FORWARD SPX RETURN')
    logger.debug('━' * 65)
    logger.debug('')
    logger.debug(f'  Step 4 OLS prediction:           {s4_current:>+.1%}  '
          f'(in-sample model, fixed betas)')
    logger.debug(f'  Step 6 Kalman raw:               {s6_pred_raw:>+.1%}  '
          f'(OOS model, time-varying betas)')
    logger.debug(f'  Step 6 Kalman bias corrected:    {s6_pred_bc:>+.1%}  '
          f'(calibrated across all regimes)')
    logger.debug('')
    logger.debug(f'  Quintile-implied return:         {q_mean_realized:>+.1%}  '
          f'(historical avg for {current_quintile_label[:10]} months since {eval_start_year})')
    logger.debug(f'  Unconditional SPX mean:          {s6_y_mean:>+.1%}  '
          f'(naive benchmark)')
    logger.debug('')
    logger.debug(f'  ▶ Direction consensus:  {consensus}')
    logger.debug(f'  ▶ {val_overlay}')

    # ----------------------------------------------------------
    # FINAL STATEMENT
    # ----------------------------------------------------------
    logger.info('')
    logger.info('=' * 65)
    logger.info('FINAL SUMMARY')
    logger.info('=' * 65)
    logger.info('')
    logger.info('Final prediction = average of:')
    logger.info('1. Bias-corrected Kalman prediction (current month)')
    logger.info('2. Historical mean realized SPX return for current Quintile')
    logger.info('')
    logger.info(f'  Prediction window:    {pred_start_date.strftime("%b %Y")} → '
          f'{pred_end_date.strftime("%b %Y")}  ({forward_months}m horizon)')
    logger.info(f'  SPX at pred start:    ${spx_at_pred_start:>8,.0f}  '
          f'({pred_start_date.strftime("%b %Y")})')
    logger.info(f'  SPX live:             ${spx_live:>8,.0f}  '
          f'({spx_live_date}'
          f'{"" if live_ok else " — fallback, yfinance unavailable"})')
    logger.info(f'  Realized so far:      {realized_so_far:>+.1%}')
    logger.info('')
    logger.info(f'  {upside_prob:.0%} odds of positive return by '
          f'{pred_end_date.strftime("%b %Y")}:   '
          f'{central_case_up:>+.1%} total  /  {remaining_upside:>+.1%} remaining')
    logger.info(f'  {downside_prob:.0%} odds of negative return by '
          f'{pred_end_date.strftime("%b %Y")}:   '
          f'{central_case_down:>+.1%} total  /  {remaining_down:>+.1%} remaining')
    logger.info(f'  ({days_remaining} days remaining in window)')
    logger.info(f'% odds above reflect empirical hit rate of {current_quintile_label.split("—")[0].strip()}. all months {eval_start_year} through present {pd.Timestamp.today().year}.')
    logger.info('')
    logger.info(f'  ▶ {quadrant_val}  |  {quadrant_macro}')
    logger.info(f'     {quadrant_label} — {quadrant_desc}')

    return {
        's4_r2'               : s4_r2,
        's4_verdict'          : s4_verdict,
        's5_zscore'           : s5_zscore,
        's5_implication'      : s5_implication,
        's6_oos_r2'           : s6_oos_r2,
        's6_directional_acc'  : s6_dir_acc,
        's6_monotonicity'     : s6_mono,
        'pred_step4'          : s4_current,
        'pred_kalman_raw'     : s6_pred_raw,
        'pred_kalman_bc'      : s6_pred_bc,
        'pred_quintile_impl'  : q_mean_realized,
        'current_quintile'    : current_quintile_label,
        'consensus'           : consensus,
        'val_overlay'         : val_overlay,
        'quadrant_val'        : quadrant_val,
        'quadrant_macro'      : quadrant_macro,
        'quadrant_label'      : quadrant_label,
        'quadrant_desc'       : quadrant_desc,
        'central_case_up'     : central_case_up,
        'central_case_down'   : central_case_down,
        'upside_prob'         : upside_prob,
        'downside_prob'       : downside_prob,
        'realized_so_far'     : realized_so_far,
        'remaining_upside'    : remaining_upside,
        'remaining_down'      : remaining_down,
        'pred_start_date'     : pred_start_date,
        'pred_end_date'       : pred_end_date,
        'spx_live'            : spx_live,
        'days_remaining'      : days_remaining,
    }


#
# ADD TO MAIN ECON SCRIPT AT STEP 7:
#

# from final_results import run_final_synthesis
#
# synthesis = run_final_synthesis(
#     step4_results     = step4_results,
#     step5_results     = valuation_results,
#     step6_results     = kalman_results,
#     forward_months    = FWD_months,
#     regression_target = REGRESSION_TARGET,
#     F_smooth          = F_smooth,
#     Lambda            = Lambda,
#     df_raw            = df,
# )

'''
Sequence of data:
1. RAW DATA → FACTORS
   many economic indicators (FRED, EIA, BEA, BLS) are compressed
   into 3 factors via the EM algorithm DFM:
   Growth, Discount, Risk_Premium
   each factor is a weighted composite of the most
   relevant indicators for that economic dimension

2. FACTORS → REGRESSION (Step 4)
   OLS regression asks: historically, when Growth/Discount/
   Risk_Premium were at level X, what did SPX return over
   the next 3 months?
   produces fixed betas and validates that the relationship
   is statistically real (p-values, R²)

3. FACTORS → WALK-FORWARD KALMAN (Step 6)
   same question as Step 4 but betas are allowed to evolve
   over time via the Kalman filter
   crucially, every prediction uses only information available
   at that point in time — genuinely out-of-sample
   produces 645 monthly predictions from 1972 to present

4. PREDICTIONS → QUINTILES
   the 645 OOS predictions are ranked from most bearish
   to most bullish and split into 5 equal buckets
   for each bucket we ask: what did SPX actually return
   in months where the model was this bullish or bearish?
   this is the empirical base rate for each regime

5. CURRENT PREDICTION → QUINTILE LOOKUP → FINAL OUTPUT
   today's factor scores go into the Kalman model
   the model outputs a predicted return (+6.0%)
   that prediction is ranked against the full history
   to find which quintile it falls in (i.e. Quintile 4)
   the final output blends the model prediction with
   the empirical base rate for that quintile:
   (+6.0% + +6.2%) / 2 = +6.1% central case
   93% probability = empirical hit rate of Quintile 4 months
   -3.4% downside = avg of the 7% of Quintile 4 months that went negative
   live SPX price pulled from Yahoo Finance to compute
   realized return since prediction start and remaining implied upside
'''