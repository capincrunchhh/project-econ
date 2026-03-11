from module9_test_new_factors import run_step9a_composite_test
from module9_added_factor_walk_forward import run_step9b_factor_rebuild
import math
import logging
logger = logging.getLogger(__name__)

'''
Important note on data leakage:
-------------------------------
This is designed to consume data leakage free inputs. Any potential data leakage stems from a prior bad input, such as module 1, 2, 3, 6, or 8. 
Those results should be free of data-leakage, therefore this is free of data-leakage. 
'''


def run_factor_addition_loop(
        step8_results,
        F_smooth,
        Lambda,
        step4_results,
        kalman_results,
        valuation_results,
        df_raw,
        df_std,
        df_std_full,
        df_ranked,
        CURRENTLY_UNUSED_COLS,
        FACTOR_COLS,
        FACTOR_NAMES,
        pca_proxies,
        REGRESSION_TARGET,
        forward_months,
        start_year,
        factor_addition_threshold = 0.20,
        oos_start_year            = 1990,
        max_factors               = 7,
):
    """
    Step 9 — Iterative Factor Addition Loop

    Runs 9a (walk-forward diagnostic) and 9b (full EM rebuild) in a loop,
    adding factors one at a time as long as:
      - 9a best candidate shows ΔOOS R² > factor_addition_threshold
      - 9b full EM rebuild confirms improvement (verdict not ❌)
    Stops when no candidate clears the threshold, 9b rejects, or
    max_factors is reached.

    Parameters
    ----------
    step8_results             : dict          — output of run_composite_factor_analysis
    F_smooth                  : pd.DataFrame  — baseline factor scores
    Lambda                    : pd.DataFrame  — baseline Lambda matrix
    step4_results             : dict          — baseline OLS results
    kalman_results            : dict          — baseline Kalman results
    valuation_results         : dict          — Gordon Growth valuation results
    df_raw                    : pd.DataFrame  — raw data
    df_std                    : pd.DataFrame  — standardized data
    df_ranked                 : pd.DataFrame  — ranked series from standardize step
    CURRENTLY_UNUSED_COLS     : list          — unused series
    FACTOR_COLS               : list of lists — series per existing factor
    FACTOR_NAMES              : list of str   — existing factor names
    pca_proxies               : dict          — {factor_name: pca_series}
    REGRESSION_TARGET         : str           — e.g. 'L5_sp500_tr_6m'
    forward_months            : int           — prediction horizon
    start_year                : int           — walk-forward start year
    factor_addition_threshold : float         — minimum ΔOOS R² to attempt 9b
    max_factors               : int           — hard cap on total factor count

    Returns
    -------
    dict with keys:
        F_smooth       : final factor scores
        Lambda         : final Lambda matrix
        step4_results  : final OLS results
        kalman_results : final Kalman results
        factor_names   : final list of factor names
        all_9b_results : dict of each round's 9b output keyed by factor name
    """

    current_F_smooth       = F_smooth
    current_Lambda         = Lambda
    current_step4_results  = step4_results
    current_kalman_results = kalman_results
    current_UNUSED_COLS    = CURRENTLY_UNUSED_COLS
    current_FACTOR_COLS    = FACTOR_COLS
    current_FACTOR_NAMES   = FACTOR_NAMES
    current_pca_proxies    = pca_proxies

    all_9b_results = {}
    factor_round   = 1

    while True:
        if len(current_FACTOR_NAMES) >= max_factors:
            logger.info(f'  Hard cap of {max_factors} factors reached. Stopping.')
            break

        n_current = len(current_FACTOR_NAMES)
        logger.info('=' * 65)
        logger.info(f'FACTOR ADDITION ROUND {factor_round}')
        logger.info(f'Current model: {n_current}-factor  {current_FACTOR_NAMES}')
        logger.info(f'Threshold:     ΔOOS R² > {factor_addition_threshold}')
        logger.info('=' * 65)

        # ─────────────────────────────────────────────────────
        # STEP 9a — walk-forward diagnostic vs current baseline
        # ─────────────────────────────────────────────────────
        step9a_results = run_step9a_composite_test(
            step8_results         = step8_results,
            F_smooth              = current_F_smooth,
            df_raw                = df_raw,
            df_std                = df_std,
            CURRENTLY_UNUSED_COLS = current_UNUSED_COLS,
            REGRESSION_TARGET     = REGRESSION_TARGET,
            forward_months        = forward_months,
            start_year            = start_year,
            step4_results         = current_step4_results,
            step5_results         = valuation_results,
            kalman_results        = current_kalman_results,
            Lambda                = current_Lambda,
            df_ranked             = df_ranked,
            factor_addition_threshold = factor_addition_threshold,
        )

        if not step9a_results:
            logger.info(f'  No candidates passed filter. Stopping at {n_current}-factor model.')
            break

        # Find best candidate by raw ΔOOS R²
        best_label     = max(step9a_results, key=lambda k: step9a_results[k]['delta_oos'])
        best_delta_oos = step9a_results[best_label]['delta_oos']
        best_delta_bc  = step9a_results[best_label]['delta_oos_bc']

        logger.info('')
        logger.info(f'  Best new factor candidate:  {best_label}')
        logger.info(f'  ΔOOS R² (raw):              {best_delta_oos:+.4f}')
        logger.info(f'  ΔOOS R² (BC):               {best_delta_bc:+.4f}')

        if best_delta_oos is None or math.isnan(best_delta_oos) or best_delta_oos < factor_addition_threshold:
            logger.info(f'  ΔOOS R² {best_delta_oos:+.4f} < threshold {factor_addition_threshold}.')
            logger.info(f'  Stopping. Final model: {n_current}-factor  {current_FACTOR_NAMES}')
            break

        logger.info(f'  Threshold cleared — proceeding to EM rebuild.')

        # ─────────────────────────────────────────────────────
        # STEP 9b — full EM rebuild with winning composite
        # ─────────────────────────────────────────────────────
        step9b_results = run_step9b_factor_rebuild(
            winning_composite_name = best_label,
            step9a_results         = step9a_results,
            df_std                 = df_std,
            df_std_full            = df_std_full,
            df_raw                 = df_raw,
            EXISTING_FACTOR_COLS   = current_FACTOR_COLS,
            EXISTING_FACTOR_NAMES  = current_FACTOR_NAMES,
            CURRENTLY_UNUSED_COLS  = current_UNUSED_COLS,
            REGRESSION_TARGET      = REGRESSION_TARGET,
            forward_months         = forward_months,
            start_year             = start_year,
            step4_results          = current_step4_results,
            kalman_results         = current_kalman_results,
            step5_results          = valuation_results,
            df_ranked              = df_ranked,
            Lambda                 = current_Lambda,
            F_smooth               = current_F_smooth,
            pca_proxies            = current_pca_proxies,
            oos_start_year         = oos_start_year,
        )

        if '❌' in step9b_results['verdict']:
            logger.info(f'  EM rebuild rejected {best_label}.')
            logger.info(f'  Stopping. Final model: {n_current}-factor  {current_FACTOR_NAMES}')
            break

        new_factor_name = step9b_results['new_factor_name']
        all_9b_results[new_factor_name] = step9b_results

        logger.info('')
        logger.info(f'  ✅ Round {factor_round} complete — {new_factor_name} added.')
        logger.info(f'  Model upgraded: {n_current}-factor → {n_current + 1}-factor')
        logger.info(f'  OOS R² (raw):   {current_kalman_results["oos_r2"]:.4f} → {step9b_results["oos_r2_nf"]:.4f}')
        logger.info(f'  OOS R² (BC):    {current_kalman_results["oos_r2_bc"]:.4f} → {step9b_results["oos_r2_bc_nf"]:.4f}')

        # ─────────────────────────────────────────────────────
        # UPDATE LOOP STATE for next round
        # ─────────────────────────────────────────────────────
        new_composite_cols = step9b_results['NEW_COMPOSITE_COLS']

        current_F_smooth       = step9b_results['F_smooth_nf']
        current_Lambda         = step9b_results['Lambda_nf']
        current_step4_results  = step9b_results['step4_nf']
        current_kalman_results = step9b_results['kalman_nf']
        current_FACTOR_COLS    = current_FACTOR_COLS + [new_composite_cols]
        current_FACTOR_NAMES   = current_FACTOR_NAMES + [new_factor_name]
        current_UNUSED_COLS    = [c for c in current_UNUSED_COLS if c not in new_composite_cols]
        current_pca_proxies    = {
            **current_pca_proxies,
            new_factor_name: step9b_results['pca_new_factor'],
        }

        factor_round += 1

    return {
        'F_smooth'       : current_F_smooth,
        'Lambda'         : current_Lambda,
        'step4_results'  : current_step4_results,
        'kalman_results' : current_kalman_results,
        'factor_names'   : current_FACTOR_NAMES,
        'all_9b_results' : all_9b_results,
    }