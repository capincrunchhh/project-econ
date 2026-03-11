import pandas as pd
import numpy as np
import logging
import time
logger = logging.getLogger(__name__)
from module3_EM_algo import run_em_dfm
from module1_data_standardize import rank_and_assign_series


'''
Important note on data leakage:
-------------------------------
    Walk-forward EM — produces a leakage-free F_smooth for use in Step 6.

    At each month t from oos_start_year onward, runs EM on data[0:t] only,
    warm-starting from the previous iteration's converged parameters.
    Records F_smooth[t-1] (the last row) as the factor reading for month t.

    This eliminates the look-ahead in the full-sample RTS smoother — each
    month's factor score is estimated using only data available up to that point.

    df_std_full is used for the burn-in period only (pre-OOS).
    Burn-in scores are never used for OOS evaluation — they only warm-start
    the walk-forward loop. Using full-sample standardization for burn-in is
    acceptable because burn-in factor scores never feed directly into predictions.

    df_std (expanding-window) is used for the walk-forward loop (OOS period).
    Each month t uses only data available up to that point — genuinely leakage-free.

    Dynamic re-ranking:
    -------------------
    At the start of each calendar year within the OOS window, rank_and_assign_series
    is re-run using only data available up to that point. This allows series that
    lacked sufficient pre-OOS history to be promoted into factor buckets once they
    accumulate enough observations — without introducing look-ahead bias.

    The number of factors (n_factors) is fixed at the burn-in value throughout.
    Only the series composition within each factor bucket changes over time.
    Lambda is re-initialized from the tiebreaker/anchor structure when all_cols changes,
    then warm-started from the previous period's converged Lambda where cols overlap.

    Bucket membership is monotonically additive — once a series is assigned to a
    bucket it stays there. Re-ranking can only add new series, never reassign existing
    ones. This prevents series from jumping between factors across years.
'''


def run_walkforward_em(
        df_std,
        df_std_full,
        all_cols,
        lambda_init,
        F0_init,
        oos_start_year  = 2000,
        n_iter          = 200,
        tol             = 1e-4,
        factor_names    = None,
        # --- dynamic re-ranking params ---
        df_raw          = None,
        target_col      = None,
        r2_threshold    = 0.05,
        pval_threshold  = 0.10,
        tiebreaker_gap  = 0.05,
        forward_months  = 1,
        # --- seed bucket lists for tracking terminal state ---
        growth_cols     = None,
        discount_cols   = None,
        risk_prem_cols  = None,
):
    """
    Parameters
    ----------
    df_std          : pd.DataFrame  — expanding-window standardized data (T x N)
                                      used for walk-forward loop (Steps 6+)
    df_std_full     : pd.DataFrame  — full-sample standardized data (T x N)
                                      used for burn-in only (pre-OOS)
    all_cols        : list          — ordered list of series fed into EM at burn-in
                                      (GROWTH_COLS_WF + DISCOUNT_COLS_WF + RISK_PREMIUM_COLS_WF)
                                      dynamically extended each year within OOS window
    lambda_init     : np.ndarray    — initial Lambda matrix from module 2 (N x n_factors)
    F0_init         : list          — initial factor state vector from module 2
    oos_start_year  : int           — first year of OOS window (burn-in = everything before)
                                      default 2000 gives ~360 months of burn-in
    n_iter          : int           — max EM iterations per window
    tol             : float         — EM convergence tolerance
    factor_names    : list|None     — column names for output DataFrame
                                      defaults to ['Growth', 'Discount', 'Risk_Premium']
    df_raw          : pd.DataFrame  — raw unstandardized data, required for dynamic re-ranking
    target_col      : str           — SPX target column, required for dynamic re-ranking
    r2_threshold    : float         — R² threshold passed to rank_and_assign_series
    pval_threshold  : float         — p-value threshold passed to rank_and_assign_series
    tiebreaker_gap  : float         — tiebreaker gap passed to rank_and_assign_series
    forward_months  : int           — forward horizon passed to rank_and_assign_series
    growth_cols     : list|None     — seed Growth bucket list, used to initialize current_G
    discount_cols   : list|None     — seed Discount bucket list, used to initialize current_D
    risk_prem_cols  : list|None     — seed Risk_Premium bucket list, used to initialize current_RP

    Returns
    -------
    F_smooth_wf : pd.DataFrame — walk-forward factor scores (T x n_factors)
                                 rows before oos_start_year are from the
                                 burn-in EM run (full burn-in window, no leakage
                                 beyond that window)
    current_G   : list — terminal Growth bucket after final re-rank
    current_D   : list — terminal Discount bucket after final re-rank
    current_RP  : list — terminal Risk_Premium bucket after final re-rank
    """

    logger.info('Creating F_smooth_wf. F_smooth contains data-leak which inflates OOS R²')
    logger.info('F_smooth_wf resolves this issue.')

    if factor_names is None:
        factor_names = ['Growth', 'Discount', 'Risk_Premium']

    dynamic_rerank = df_raw is not None and target_col is not None

    if dynamic_rerank:
        logger.info('Dynamic re-ranking enabled — bucket lists will update annually within OOS window')
    else:
        logger.info('Dynamic re-ranking disabled — using static bucket lists throughout')

    # Initialize terminal bucket trackers from seed lists.
    # These are extended (never replaced) each time a re-rank fires.
    # Returned at end so main has the final terminal state for Steps 8-9.
    current_G  = list(growth_cols)    if growth_cols    else []
    current_D  = list(discount_cols)  if discount_cols  else []
    current_RP = list(risk_prem_cols) if risk_prem_cols else []

    n_factors  = lambda_init.shape[1]
    dates      = df_std.index
    T          = len(dates)

    # Compute oos_idx before using it
    oos_start  = pd.Timestamp(f'{oos_start_year}-01-01')
    oos_idx    = int(np.searchsorted(dates, oos_start))

    # Full-sample data — used for burn-in only (has way less NaNs, burn-in scores never
    # used for prediction so using df_std_full here doesn't contaminate OOS R2)
    Y_burnin   = df_std_full[all_cols].values[:oos_idx].astype(float)

    logger.debug('=' * 65)
    logger.debug('Walk-Forward EM — Leakage-Free Factor Estimation')
    logger.debug('=' * 65)
    logger.debug(f'  Burn-in window:   1970-01 → {dates[oos_idx - 1].strftime("%Y-%m")}  ({oos_idx} months)')
    logger.debug(f'  OOS window:       {dates[oos_idx].strftime("%Y-%m")} → {dates[-1].strftime("%Y-%m")}  ({T - oos_idx} months)')
    logger.debug(f'  Total iterations: {T - oos_idx}')
    logger.debug(f'  Series:           {len(all_cols)}')
    logger.debug(f'  Factors:          {n_factors}')
    logger.debug('')

    # Output array — fill with NaN, populate as we go
    F_smooth_wf = np.full((T, n_factors), np.nan)

    # Suppress EM and factor bucket assignment logging during walk-forward loop
    em_logger          = logging.getLogger('module3_EM_algo')
    standardize_logger = logging.getLogger('module1_data_standardize')
    em_logger.setLevel(logging.WARNING)
    standardize_logger.setLevel(logging.WARNING)

    # --- Burn-in run ---
    # Run full EM on full-sample data[0:oos_idx] to get starting params
    # and factor scores for the pre-OOS period.
    # Uses df_std_full (not df_std_wf) to avoid NaN-heavy early rows.
    logger.debug(f'  Running burn-in EM on {oos_idx} months...')

    burnin_results = run_em_dfm(
        Y           = Y_burnin,
        lambda_init = lambda_init,
        F0          = F0_init,
        n_iter      = n_iter,
        tol         = tol,
    )

    # Store burn-in factor scores
    F_smooth_wf[:oos_idx] = burnin_results['F_smooth']

    # Warm-start params for first OOS iteration
    prev_Lambda = burnin_results['Lambda']
    prev_A      = burnin_results['A']
    prev_Q      = burnin_results['Q']
    prev_R      = burnin_results['R']
    prev_F0     = burnin_results['F0']
    prev_P0     = burnin_results['P0']

    # Track current flat col list so we can detect changes after re-ranking
    current_all_cols = list(all_cols)

    logger.debug(f'  Burn-in complete. Beginning walk-forward...')
    logger.debug('')

    loop_start = time.time()

    # Pre-compute hybrid Y slices for the walk-forward loop:
    # Pre-OOS rows from df_std_full (NaN-safe), OOS rows from df_std_wf (leakage-free)
    Y_pre_oos = df_std_full[all_cols].values[:oos_idx].astype(float)
    Y_oos_wf  = df_std[all_cols].values[oos_idx:].astype(float)

    last_rerank_year = None

    # --- Walk-forward loop ---
    # Each step t uses expanding-window standardized data[0:t] — leakage-free
    for i, t in enumerate(range(oos_idx, T)):

        current_date = dates[t]
        current_year = current_date.year

        # --- Dynamic re-ranking: once per calendar year ---
        if dynamic_rerank and current_year != last_rerank_year:

            G, D, RP, _, _ = rank_and_assign_series(
                df             = df_raw,
                target_col     = target_col,
                forward_months = forward_months,
                r2_threshold   = r2_threshold,
                pval_threshold = pval_threshold,
                tiebreaker_gap = tiebreaker_gap,
                oos_start_year = current_year,   # only data available up to now
            )

            # Monotonic promotion only — series already assigned stay in their bucket.
            # Re-ranking can only add new series, never reassign existing ones.
            # This prevents series from jumping between factors across years.
            all_currently_assigned = set(current_G) | set(current_D) | set(current_RP)

            newly_added = []
            for col in G:
                if col not in all_currently_assigned:
                    current_G.append(col)
                    newly_added.append(col)
                    all_currently_assigned.add(col)
            for col in D:
                if col not in all_currently_assigned:
                    current_D.append(col)
                    newly_added.append(col)
                    all_currently_assigned.add(col)
            for col in RP:
                if col not in all_currently_assigned:
                    current_RP.append(col)
                    newly_added.append(col)
                    all_currently_assigned.add(col)

            new_all_cols = current_G + current_D + current_RP

            if newly_added:
                logger.info(
                    f'  [{current_date.strftime("%Y")}] Re-rank: '
                    f'+{len(newly_added)} series promoted  '
                    f'(total # series incl in factors: {len(new_all_cols)})'
                )
                logger.debug(f'    Promoted: {sorted(newly_added)}')

                # Rebuild Y slices with updated col list
                Y_pre_oos = df_std_full[new_all_cols].values[:oos_idx].astype(float)
                Y_oos_wf  = df_std[new_all_cols].values[oos_idx:].astype(float)

                # Carry over Lambda rows for existing cols; re-init new ones to zero
                new_n       = len(new_all_cols)
                new_Lambda  = np.zeros((new_n, n_factors))
                old_col_idx = {c: idx for idx, c in enumerate(current_all_cols)}
                for new_i, col in enumerate(new_all_cols):
                    if col in old_col_idx:
                        new_Lambda[new_i] = prev_Lambda[old_col_idx[col]]
                    # new series start at zero — EM will estimate from data

                prev_Lambda      = new_Lambda
                current_all_cols = new_all_cols

            last_rerank_year = current_year

        # Hybrid expanding window: more robust starting pre-OOS rows + leakage-free OOS rows thereafter
        Y_t = np.vstack([Y_pre_oos, Y_oos_wf[:t - oos_idx]])

        results_t = run_em_dfm(
            Y           = Y_t,
            lambda_init = prev_Lambda,
            F0          = list(prev_F0),
            n_iter      = n_iter,
            tol         = tol,
        )

        # Record the last row of F_smooth as today's factor reading
        # F_smooth[-1] = estimate for month t-1 using data[0:t]
        F_smooth_wf[t] = results_t['F_smooth'][-1]

        # Update warm-start params for next iteration
        prev_Lambda = results_t['Lambda']
        prev_A      = results_t['A']
        prev_Q      = results_t['Q']
        prev_R      = results_t['R']
        prev_F0     = results_t['F0']
        prev_P0     = results_t['P0']

        # Progress logging every 50 months
        if i == 0 or (i + 1) % 50 == 0 or t == T - 1:
            elapsed   = time.time() - loop_start
            pct_done  = (i + 1) / (T - oos_idx)
            eta       = (elapsed / pct_done) - elapsed if pct_done > 0 else 0
            logger.warning(
                f'  EM walk-fwd progress: {i + 1}/{T - oos_idx} months  '
                f'({pct_done:.0%})  eta={eta / 60:.1f} min remaining'
            )

    # Restore logging after loop
    em_logger.setLevel(logging.INFO)
    standardize_logger.setLevel(logging.INFO)

    total_time = time.time() - loop_start
    logger.debug('')
    logger.debug(f'  Walk-forward complete.  Total time: {total_time:.1f}s')

    F_smooth_wf = pd.DataFrame(
        F_smooth_wf,
        index   = dates,
        columns = factor_names,
    )

    return F_smooth_wf, current_G, current_D, current_RP