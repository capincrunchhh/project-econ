import pandas as pd
import numpy as np
import statsmodels.api as sm
from module6_walk_forward_optimization import run_kalman_regression
from module8_added_factors import build_equal_weight_composite, build_pca_composite
import logging
logger = logging.getLogger(__name__)


'''
Important note on data leakage:
-------------------------------
This is designed to consume data leakage free inputs. Any potential data leakage stems from a prior bad input, such as module 1, 2, 3, 6, or 8. 
Those results should be free of data-leakage, therefore this is free of data-leakage. 
'''


def run_step9a_composite_test(
        step8_results,
        F_smooth,
        df_raw,
        df_std,
        CURRENTLY_UNUSED_COLS,
        REGRESSION_TARGET,
        forward_months,
        start_year,
        step4_results,
        step5_results,
        kalman_results,
        Lambda,
        df_ranked,
        beta_drift_q              = 0.001,
        obs_noise_r               = None,
        factor_addition_threshold = 0.10,
):
    """
    Step 9a — 4th Factor Walk-Forward Diagnostic

    Takes top-ranked composites from Step 8 and appends each as a 4th
    factor to F_smooth. Re-runs Steps 4 and 6 for each candidate and
    compares OOS R² against the 3-factor baseline from kalman_results.

    No EM re-run — composite scores are pre-built exogenous regressors.
    If any candidate improves OOS R² meaningfully, flags for Step 9b
    (full EM rebuild with 4th factor properly integrated).

    Publication lags handled via forward-fill — last valid composite
    value carried forward to current month. This is consistent with
    how the 3-factor model handles slow-moving series and is the
    operationally correct approach for production use.

    Parameters
    ----------
    step8_results             : dict          — output of run_composite_factor_analysis
    F_smooth                  : pd.DataFrame  — 3-factor Kalman scores (T x 3)
    df_raw                    : pd.DataFrame  — raw unstandardized data
    df_std                    : pd.DataFrame  — standardized data
    CURRENTLY_UNUSED_COLS     : list          — unused series from 3-factor model
    REGRESSION_TARGET         : str           — e.g. 'L5_sp500_tr_6m'
    forward_months            : int           — prediction horizon
    start_year                : int           — walk-forward start year
    step4_results             : dict          — 3-factor OLS results (baseline)
    step5_results             : dict          — Gordon Growth valuation results
    kalman_results            : dict          — 3-factor Kalman results (baseline OOS R²)
    Lambda                    : pd.DataFrame  — 3-factor Lambda matrix
    df_ranked                 : pd.DataFrame  — ranked series from standardize step
    beta_drift_q              : float         — Kalman beta drift parameter
    obs_noise_r               : float|None    — Kalman observation noise
    factor_addition_threshold : float         — minimum ΔOOS R² to recommend Step 9b
                                               (passed from econ_model.py FACTOR_ADDITION_THRESHOLD)

    Returns
    -------
    results : dict — per-candidate OOS R² comparison and full Kalman output
    """

    # --- 3-factor baselines ---
    baseline_is_r2     = step4_results['model'].rsquared
    baseline_oos_r2    = kalman_results['oos_r2']
    baseline_oos_r2_bc = kalman_results['oos_r2_bc']

    # --- Rank candidates by incremental in-sample R² from Step 8 ---
    # Only test composites that showed meaningful incremental signal
    candidates = []
    for name, res in step8_results.items():
        for method, stats, composite in [
            ('EW',  res['ew_stats'],  res['ew_composite']),
            ('PCA', res['pca_stats'], res['pca_composite']),
        ]:
            if stats is None or composite is None:
                continue
            if stats['r2_delta'] > 0.01 and stats['pval_incr'] < 0.05:
                candidates.append({
                    'label'     : f'{name}_{method}',
                    'composite' : composite,
                    'r2_delta'  : stats['r2_delta'],
                    'pval_incr' : stats['pval_incr'],
                })

    candidates = sorted(candidates, key=lambda x: x['r2_delta'], reverse=True)

    # --- Deduplicate by composite group — keep best ΔIS R² per family ---
    # Sorted by r2_delta descending above, so first seen per group is the best.
    # Explicit map handles multi-word group names (e.g. Financial_Stress).
    GROUP_MAP = {
        'Financial_Stress_EW':  'Financial_Stress',
        'Financial_Stress_PCA': 'Financial_Stress',
        'Labor_EW':             'Labor',
        'Labor_PCA':            'Labor',
        'Consumer_EW':          'Consumer',
        'Consumer_PCA':         'Consumer',
        'Corporate_EW':         'Corporate',
        'Corporate_PCA':        'Corporate',
        'Government_EW':        'Government',
        'Government_PCA':       'Government',
    }

    seen_groups = {}
    deduped     = []
    dropped     = []
    for c in candidates:
        group = GROUP_MAP.get(c['label'], c['label'])
        if group not in seen_groups:
            seen_groups[group] = c['label']
            deduped.append(c)
        else:
            dropped.append((c['label'], group, seen_groups[group]))

    candidates = deduped

    # =========================================================
    # HEADER
    # =========================================================
    logger.info('=' * 65)
    n_existing = len(F_smooth.columns)
    n_new = n_existing + 1
    logger.info(f'{n_existing + 1}th FACTOR WALK-FORWARD DIAGNOSTIC')
    logger.info('=' * 65)
    logger.info(f'  {n_existing}-factor baseline in-sample R²:   {baseline_is_r2:.4f}')
    logger.info(f'  {n_existing}-factor baseline OOS R²:         {baseline_oos_r2:.4f}')
    logger.info(f'  {n_existing}-factor baseline OOS R² (BC):    {baseline_oos_r2_bc:.4f}')
    logger.info(f'  New factors to test:               {len(candidates)}')
    logger.info(f'  Inclusion threshold:                ΔOOS R² > {factor_addition_threshold:.2f}')
    if dropped:
        logger.debug('')
        logger.debug(f'  [Deduplication] One candidate kept per composite family (best ΔIS R²):')
        for label, group, kept in dropped:
            logger.debug(f'    Dropped {label:<30}  group={group},  kept={kept}')
    logger.debug('')
    if candidates:
        logger.info(f'  Ranked by in-sample incremental R²:')
        for c in candidates:
            logger.info(f'    {c["label"]:<35}  ΔR²={c["r2_delta"]:+.4f}  p={c["pval_incr"]:.4f}')
    else:
        logger.info('  No candidates passed filter (ΔR² > 0.01, p < .05)')
        logger.info('  Nothing to test. Retain 3-factor model.')
        return {}

    results = {}

    for c in candidates:

        label     = c['label']
        composite = c['composite']

        logger.debug('')
        logger.debug('=' * 65)
        logger.debug(f'  TESTING: {label}')
        logger.debug('=' * 65)

        # --- Build 4-factor F_smooth ---
        F_4factor        = F_smooth.copy()
        F_4factor[label] = composite

        # --- Forward-fill to handle publication lags ---
        last_valid = F_4factor[label].last_valid_index()
        n_filled   = F_4factor[label].isna().sum()
        F_4factor[label] = F_4factor[label].ffill()

        n_total     = len(F_4factor)
        first_valid = F_4factor[label].first_valid_index()
        first_str   = first_valid.strftime('%Y-%m') if first_valid is not None else 'N/A'
        last_str    = last_valid.strftime('%Y-%m') if last_valid is not None else 'N/A'

        logger.debug('')
        logger.debug(f'  Composite history:  {first_str} → {F_4factor.index[-1].strftime("%Y-%m")}')
        logger.debug(f'  Last data point:    {last_str}  (forward-filled {n_filled} month(s) to current)')
        logger.debug(f'  Valid observations: {n_total} of {n_total} months (after ffill)')
        logger.debug('')

        # =========================================================
        # RE-RUN STEP 4 (OLS) WITH 4 FACTORS
        # =========================================================
        from module4_spx_regression import run_spx_regression

        logger.debug('  ' + '─' * 65)
        logger.debug('  Re-run OLS with 4-factors')
        logger.debug('  ' + '─' * 65)

        step4_4f = run_spx_regression(
            factors        = F_4factor,
            spx            = df_raw[[REGRESSION_TARGET]],
            forward_months = forward_months,
        )

        # =========================================================
        # RE-RUN STEP 6 (KALMAN) WITH 4 FACTORS
        # =========================================================
        logger.debug('')
        logger.debug('  ' + '─' * 65)
        logger.debug('  Re-run Kalman walk-forward with 4-factors')
        logger.debug('  ' + '─' * 65)

        kalman_4f = run_kalman_regression(
            F_smooth          = F_4factor,
            df_raw            = df_raw,
            REGRESSION_TARGET = REGRESSION_TARGET,
            forward_months    = forward_months,
            start_year        = start_year,
            beta_drift_q      = beta_drift_q,
            obs_noise_r       = obs_noise_r,
            in_sample_r2      = step4_4f['model'].rsquared,
            enable_graph      = False,
        )

        # =========================================================
        # COMPARISON TABLE
        # =========================================================
        oos_r2_4f    = kalman_4f['oos_r2']
        oos_r2_bc_4f = kalman_4f['oos_r2_bc']
        is_r2_4f     = step4_4f['model'].rsquared

        delta_is     = is_r2_4f     - baseline_is_r2
        delta_oos    = oos_r2_4f    - baseline_oos_r2
        delta_oos_bc = oos_r2_bc_4f - baseline_oos_r2_bc

        if delta_oos > 0.010:
            verdict = '✅  ELIGIBLE'
        elif delta_oos > 0.000:
            verdict = '⚠️  MARGINAL'
        else:
            verdict = '❌  EXCLUDE'

        # Dynamically size box to fit the widest content
        METRIC_W = 30
        COL_W    = 9
        fixed_inner = 2 + METRIC_W + 1 + COL_W + 1 + COL_W + 2

        header_text  = f'COMPARISON: {n_existing}-factor vs {n_new}-factor [{label}]'
        verdict_text = f'{"Verdict":<{METRIC_W}} {"":>{COL_W}} {verdict}'
        inner_w      = max(fixed_inner, len(header_text) + 4, len(verdict_text) + 4)

        bar = '─' * inner_w
        top = '┌' + bar + '┐'
        mid = '├' + bar + '┤'
        bot = '└' + bar + '┘'

        def data_row(metric, v3='', v4=''):
            content = f'  {metric:<{METRIC_W}} {str(v3):>{COL_W}} {str(v4):>{COL_W}}  '
            return f'  │{content:<{inner_w}}│'

        def hdr_row(text):
            content = f'  {text}'
            return f'  │{content:<{inner_w}}│'

        logger.debug('')
        logger.debug(f'  {top}')
        logger.debug(hdr_row(header_text))
        logger.debug(f'  {mid}')
        logger.debug(data_row('Metric', f'{n_existing}-factor', f'{n_new}-factor'))
        logger.debug(f'  {mid}')
        logger.debug(data_row('In-sample R²',            f'{baseline_is_r2:.4f}',                    f'{is_r2_4f:.4f}'))
        logger.debug(data_row('OOS R² (raw)',             f'{baseline_oos_r2:.4f}',                   f'{oos_r2_4f:.4f}'))
        logger.debug(data_row('OOS R² (bias corrected)',  f'{baseline_oos_r2_bc:.4f}',                f'{oos_r2_bc_4f:.4f}'))
        logger.debug(data_row('Directional accuracy',     f'{kalman_results["directional_acc"]:.1%}', f'{kalman_4f["directional_acc"]:.1%}'))
        logger.debug(data_row('Directional acc (BC)',     f'{kalman_results["dir_acc_bc"]:.1%}',      f'{kalman_4f["dir_acc_bc"]:.1%}'))
        logger.debug(data_row('Monotonicity',             f'{kalman_results["monotonicity"]:.0%}',    f'{kalman_4f["monotonicity"]:.0%}'))
        logger.debug(data_row('MAE',                      f'{kalman_results["mae"]:.1%}',             f'{kalman_4f["mae"]:.1%}'))
        logger.debug(f'  {mid}')
        logger.debug(data_row('ΔIn-sample R²',           '', f'{delta_is:+.4f}'))
        logger.debug(data_row('ΔOOS R² (raw)',            '', f'{delta_oos:+.4f}'))
        logger.debug(data_row('ΔOOS R² (bias corrected)', '', f'{delta_oos_bc:+.4f}'))
        logger.debug(f'  {mid}')
        logger.debug(data_row('Verdict', '', verdict))
        logger.debug(f'  {bot}')

        results[label] = {
            'step4_4f'     : step4_4f,
            'kalman_4f'    : kalman_4f,
            'F_4factor'    : F_4factor,
            'is_r2_4f'     : is_r2_4f,
            'oos_r2_4f'    : oos_r2_4f,
            'oos_r2_bc_4f' : oos_r2_bc_4f,
            'delta_is'     : delta_is,
            'delta_oos'    : delta_oos,
            'delta_oos_bc' : delta_oos_bc,
            'verdict'      : verdict,
            'last_valid'   : last_valid,
            'n_filled'     : n_filled,
        }

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    logger.info('=' * 65)
    logger.info(f'{n_existing + 1}th FACTOR WALK-FORWARD DIAGNOSTIC SUMMARY')
    logger.info('=' * 65)
    logger.info(f'  {"Composite":<35} {"ΔIS R²":>8} {"ΔOOS R²":>8} {"ΔOOS BC":>8}  {"Last Data":>10}  Verdict')
    logger.info(f'  {"─" * 85}')

    best_label     = None
    best_delta_oos = -999

    for label, res in results.items():
        last_str = res['last_valid'].strftime('%Y-%m') if res['last_valid'] is not None else 'N/A'
        logger.info(f'  {label:<35}'
              f' {res["delta_is"]:>+8.4f}'
              f' {res["delta_oos"]:>+8.4f}'
              f' {res["delta_oos_bc"]:>+8.4f}'
              f'  {last_str:>10}'
              f'  {res["verdict"]}')
        if res['delta_oos'] > best_delta_oos:
            best_delta_oos = res['delta_oos']
            best_label     = label

    logger.info('')
    logger.info(f'  Baseline OOS R²:       {baseline_oos_r2:.4f}')
    logger.info(f'  Baseline OOS R² (BC):  {baseline_oos_r2_bc:.4f}')
    logger.info('')

    if best_label and results[best_label]['delta_oos'] > factor_addition_threshold:
        logger.info(f'  ✅ Best candidate:  {best_label}')
        logger.debug(f'     ΔOOS R²:        {results[best_label]["delta_oos"]:+.4f}')
        logger.debug(f'     ΔOOS R² (BC):   {results[best_label]["delta_oos_bc"]:+.4f}')
        logger.debug(f'     Last data:      {results[best_label]["last_valid"].strftime("%Y-%m")}')
        logger.debug('')
        logger.info(f'  → Proceed to: full EM rebuild with {best_label} as {n_existing + 1}th factor')
    else:
        logger.info(f'  ❌ No composite improved OOS R² by more than {factor_addition_threshold:.2f}')
        logger.debug(f'     Best result: {best_label} at ΔOOS R²={best_delta_oos:+.4f}')
        logger.debug('')
        logger.info(f'  → Retain {n_existing}-factor model. Next step not required.')

    logger.info('')

    return results