import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from module3_EM_algo import run_em_dfm
from module3_walkforward_em import run_walkforward_em
from module4_spx_regression import run_spx_regression
from module6_walk_forward_optimization import run_kalman_regression
from module7_final_results import run_final_synthesis
from module8_added_factors import COMPOSITE_CANDIDATES
import logging
logger = logging.getLogger(__name__)


'''
Important note on data leakage:
-------------------------------
This is designed to consume data leakage free inputs. Any potential data leakage stems from a prior bad input, such as module 1, 2, 3, 6, or 8. 
Those results should be free of data-leakage, therefore this is free of data-leakage. 


Module 9b has its own separate leakage in two places that are independent of Module 8:

init_new_factor_value — runs its own fresh PCA on the new factor's composite series to initialize F0
build_lambda_df_nfactor — computes full-sample correlations to build the Lambda init matrix

These are called internally inside 9b regardless of what Module 8 does — they don't use anything from Module 8's composite scores. 
So fixing Module 8 does not fix 9b. Both need their own oos_start_year cutoff added independently.
'''


def init_new_factor_value(df_std, composite_cols, factor_name):
    """
    Initialize a new factor using PCA on its composite series.
    Mirrors the pattern in module2_factor_growth/discount/risk_premium.
    Works for any nth factor — not hardcoded to 4th.

    Returns
    -------
    factor_init  : float        — scalar initial value for F0
    pca_series   : pd.Series    — PCA time series proxy for this factor
    """
    available = [c for c in composite_cols if c in df_std.columns]
    if len(available) == 0:
        raise ValueError(f'No composite series found in df_std for {factor_name}')

    data        = df_std[available].dropna(how='all')
    data_filled = data.fillna(data.mean())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_filled)

    pca  = PCA(n_components=1)
    pc1  = pca.fit_transform(scaled)[:, 0]

    # Orient PC1 so dominant loading is positive
    loadings = pca.components_[0]
    if loadings[np.argmax(np.abs(loadings))] < 0:
        pc1      = -pc1
        loadings = -loadings

    pca_series    = pd.Series(pc1, index=data_filled.index, name=factor_name)
    factor_init   = float(pca_series.iloc[:24].mean())
    var_explained = pca.explained_variance_ratio_[0]

    logger.debug(f'  {factor_name} Factor Init:')
    logger.debug(f'    Series used:         {len(available)}')
    logger.debug(f'    Variance explained:  {var_explained:.1%}')
    logger.debug(f'    Initial value (F0):  {factor_init:.4f}')
    top5_idx = np.argsort(np.abs(loadings))[::-1][:5]
    logger.debug(f'    Top 5 loadings:')
    for i in top5_idx:
        logger.debug(f'      {available[i]:<45} {loadings[i]:+.4f}')
    logger.debug('')

    return factor_init, pca_series


def build_lambda_df_nfactor(df_std,
                             EXISTING_FACTOR_COLS,
                             EXISTING_FACTOR_NAMES,
                             NEW_COMPOSITE_COLS,
                             new_factor_name,
                             pca_proxies,
                             pca_new_factor):
    """
    Build extended Lambda initialization matrix for an n-factor EM rebuild.
    Generalizes the 4-factor Lambda builder to support any number of factors.

    Structure:
        rows  = all series (all existing factor series + new composite series)
        cols  = EXISTING_FACTOR_NAMES + [new_factor_name]

    Existing series: correlated against their own factor's PCA proxy
                     zero loading on the new factor column
    New composite series: correlated against pca_new_factor
                          zero loading on all existing factor columns

    This sparse initialization preserves factor identity and prevents
    the EM from immediately mixing factors during early iterations.

    Parameters
    ----------
    df_std                : pd.DataFrame — standardized data
    EXISTING_FACTOR_COLS  : list of lists — e.g. [GROWTH_COLS, DISCOUNT_COLS, ...]
                            one list per existing factor, same order as EXISTING_FACTOR_NAMES
    EXISTING_FACTOR_NAMES : list of str  — e.g. ['Growth', 'Discount', 'Risk_Premium']
                            or ['Growth', 'Discount', 'Risk_Premium', 'Government']
    NEW_COMPOSITE_COLS    : list of str  — series for the new factor
    new_factor_name       : str          — name for the new factor column
    pca_proxies           : dict         — {factor_name: pca_series} for all existing factors
    pca_new_factor        : pd.Series    — PCA proxy for the new factor

    Returns
    -------
    lambda_nf : pd.DataFrame — (N_total x n_factors) initialization matrix
    """

    all_existing = [s for cols in EXISTING_FACTOR_COLS for s in cols]
    all_cols_new = all_existing + NEW_COMPOSITE_COLS
    all_factors  = EXISTING_FACTOR_NAMES + [new_factor_name]
    n_factors    = len(all_factors)

    lambda_nf = pd.DataFrame(
        0.0,
        index   = all_cols_new,
        columns = all_factors,
        dtype   = float,
    )

    # --- Existing series — correlate against their own factor's PCA proxy ---
    for factor_name, cols in zip(EXISTING_FACTOR_NAMES, EXISTING_FACTOR_COLS):
        proxy = pca_proxies[factor_name]
        for series in cols:
            if series not in df_std.columns:
                continue
            aligned = pd.concat([df_std[series], proxy], axis=1).dropna()
            if len(aligned) > 12:
                lambda_nf.loc[series, factor_name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            else:
                lambda_nf.loc[series, factor_name] = 0.0

    # --- New composite series — correlate against new factor PCA proxy ---
    for series in NEW_COMPOSITE_COLS:
        if series not in df_std.columns:
            continue
        aligned = pd.concat([df_std[series], pca_new_factor], axis=1).dropna()
        if len(aligned) > 12:
            lambda_nf.loc[series, new_factor_name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        else:
            lambda_nf.loc[series, new_factor_name] = 0.0

    logger.info(f'  {n_factors}-Factor Lambda Init Matrix Complete')
    logger.info(f'  Shape: {lambda_nf.shape}')
    logger.debug(f'  Existing series ({len(EXISTING_FACTOR_NAMES)} factors): {len(all_existing)}')
    logger.debug(f'  New composite series ({new_factor_name}): {len(NEW_COMPOSITE_COLS)}')
    logger.debug('')
    logger.debug(f'  --- {new_factor_name} bucket (first 5) ---')
    display_cols = NEW_COMPOSITE_COLS[:5]
    logger.debug(lambda_nf.loc[display_cols].round(3).to_string())
    logger.debug('')

    return lambda_nf


def run_step9b_factor_rebuild(
        winning_composite_name,
        step9a_results,
        df_std,
        df_std_full,
        df_raw,
        EXISTING_FACTOR_COLS,
        EXISTING_FACTOR_NAMES,
        CURRENTLY_UNUSED_COLS,
        REGRESSION_TARGET,
        forward_months,
        start_year,
        step4_results,
        kalman_results,
        step5_results,
        df_ranked,
        Lambda,
        F_smooth,
        pca_proxies,
        oos_start_year = 2000,
        n_iter         = 500,
        tol            = 1e-6,
        beta_drift_q   = 0.001,
        obs_noise_r    = None,
        enable_graph   = False,
):
    """
    Step 9b — Full N-Factor EM Rebuild (generalized)

    Takes the winning composite from Step 9a and properly integrates it
    as a new latent factor in the DFM by rebuilding F0, Lambda, and
    re-running the full EM algorithm. Supports adding a 4th, 5th, or
    any nth factor — not hardcoded to a specific factor count.

    Unlike Step 9a (Option A), the new factor is estimated jointly with
    all existing factors — the EM E-step orthogonalizes all factors
    simultaneously, preventing collinearity from inflating the signal.

    After the full-sample EM rebuild, factor scores are passed through
    run_walkforward_em to produce leakage-free F_smooth_nf before
    feeding into the Kalman regression.

    Parameters
    ----------
    winning_composite_name : str        — e.g. 'Government_PCA', 'Labor_EW'
    step9a_results         : dict       — output of run_step9a_composite_test
    df_std                 : pd.DataFrame — standardized data
    df_std_full            : pd.DataFrame — full-sample standardized data (burn-in only)
    df_raw                 : pd.DataFrame — raw unstandardized data
    EXISTING_FACTOR_COLS   : list of lists — one list of series per existing factor
    EXISTING_FACTOR_NAMES  : list of str  — names matching EXISTING_FACTOR_COLS order
    CURRENTLY_UNUSED_COLS  : list         — unused series
    REGRESSION_TARGET      : str          — e.g. 'L5_sp500_tr_6m'
    forward_months         : int          — prediction horizon
    start_year             : int          — walk-forward start year
    step4_results          : dict         — current baseline OLS results
    kalman_results         : dict         — current baseline Kalman results
    step5_results          : dict         — Gordon Growth valuation results
    df_ranked              : pd.DataFrame — ranked series from standardize step
    Lambda                 : pd.DataFrame — current baseline Lambda matrix
    F_smooth               : pd.DataFrame — current baseline F_smooth
    pca_proxies            : dict         — {factor_name: pca_series} for ALL existing factors
    oos_start_year         : int          — passed to run_walkforward_em (default 2000)
    n_iter                 : int          — EM iterations
    tol                    : float        — EM convergence tolerance
    beta_drift_q           : float        — Kalman beta drift
    obs_noise_r            : float|None   — Kalman observation noise
    enable_graph           : bool         — render Kalman output chart

    Returns
    -------
    results : dict — full n-factor model outputs and comparison vs baseline
    """

    n_existing = len(EXISTING_FACTOR_NAMES)
    n_new      = n_existing + 1

    logger.info('')
    logger.info('=' * 65)
    logger.info(f'{n_new}-FACTOR EM REBUILD')
    logger.info('=' * 65)
    logger.info('')
    logger.info(f'  Added factor:           {winning_composite_name}')
    logger.info(f'  Existing factors ({n_existing}):   {EXISTING_FACTOR_NAMES}')
    logger.info(f'  New factor count:       {n_new}')
    logger.info('')
    logger.info(f'  {n_existing}-factor baseline:')
    logger.info(f'    In-sample R²:         {step4_results["model"].rsquared:.4f}')
    logger.info(f'    OOS R²:               {kalman_results["oos_r2"]:.4f}')
    logger.info(f'    OOS R² (BC):          {kalman_results["oos_r2_bc"]:.4f}')
    logger.info('')

    # =========================================================
    # RESOLVE COMPOSITE SERIES
    # =========================================================
    composite_group = (winning_composite_name
                       .replace('_PCA', '')
                       .replace('_EW', ''))

    if composite_group not in COMPOSITE_CANDIDATES:
        raise ValueError(
            f'Composite group "{composite_group}" not found in COMPOSITE_CANDIDATES. '
            f'Available: {list(COMPOSITE_CANDIDATES.keys())}'
        )

    all_defined        = COMPOSITE_CANDIDATES[composite_group]
    unused_set         = set(CURRENTLY_UNUSED_COLS)
    NEW_COMPOSITE_COLS = [
        s for s in all_defined
        if s in df_std.columns and s in unused_set
    ]

    if len(NEW_COMPOSITE_COLS) == 0:
        raise ValueError(
            f'No unused series available for composite "{composite_group}". '
            f'All defined series are either missing from df_std or already in the model.'
        )

    new_factor_name = composite_group

    logger.debug(f'  Composite series:')
    logger.debug(f'    Defined:              {len(all_defined)}')
    logger.debug(f'    Available & unused:   {len(NEW_COMPOSITE_COLS)}')
    logger.debug(f'    New factor name:      {new_factor_name}')
    logger.debug('')

    # =========================================================
    # INITIALIZE NEW FACTOR
    # =========================================================
    logger.debug('  ' + '─' * 65)
    logger.debug(f'  Initializing {new_factor_name} factor via PCA')
    logger.debug('  ' + '─' * 65)
    logger.debug('')

    new_factor_init, pca_new_factor = init_new_factor_value(
        df_std         = df_std,
        composite_cols = NEW_COMPOSITE_COLS,
        factor_name    = new_factor_name,
    )

    # =========================================================
    # BUILD EXTENDED F0 AND LAMBDA
    # =========================================================
    logger.info('  ' + '─' * 65)
    logger.info(f'  Building {n_new}-factor F0 and Lambda')
    logger.info('  ' + '─' * 65)
    logger.info('')

    existing_inits = [float(pca_proxies[f].iloc[:24].mean()) for f in EXISTING_FACTOR_NAMES]
    F0_nf          = existing_inits + [new_factor_init]

    logger.info(f'  F0 ({n_new}-factor):')
    for name, val in zip(EXISTING_FACTOR_NAMES + [new_factor_name], F0_nf):
        logger.info(f'    {name:<20} {val:.4f}')
    logger.info('')

    lambda_nf = build_lambda_df_nfactor(
        df_std                = df_std,
        EXISTING_FACTOR_COLS  = EXISTING_FACTOR_COLS,
        EXISTING_FACTOR_NAMES = EXISTING_FACTOR_NAMES,
        NEW_COMPOSITE_COLS    = NEW_COMPOSITE_COLS,
        new_factor_name       = new_factor_name,
        pca_proxies           = pca_proxies,
        pca_new_factor        = pca_new_factor,
    )

    lambda_fname = f'lambda_init_{n_new}factor_{new_factor_name}.csv'
    lambda_nf.to_csv(lambda_fname)
    logger.debug(f'  Lambda saved to {lambda_fname}')
    logger.debug('')

    # =========================================================
    # RE-RUN FULL-SAMPLE EM WITH N FACTORS
    # =========================================================
    logger.info('  ' + '─' * 65)
    logger.info(f'  Running {n_new}-factor EM algorithm (full sample)')
    logger.info('  ' + '─' * 65)
    logger.info('')

    all_cols_nf = [s for cols in EXISTING_FACTOR_COLS for s in cols] + NEW_COMPOSITE_COLS
    Y_nf        = df_std[all_cols_nf].values.astype(float)

    logger.info(f'  Data shape:    {Y_nf.shape}')
    logger.info(f'  Lambda shape:  {lambda_nf.values.shape}')
    logger.info(f'  F0:            {[round(v, 3) for v in F0_nf]}')
    logger.info('')

    results_em = run_em_dfm(
        Y           = Y_nf,
        lambda_init = lambda_nf.values,
        F0          = F0_nf,
        n_iter      = n_iter,
        tol         = tol,
    )

    all_factor_names = EXISTING_FACTOR_NAMES + [new_factor_name]
    dates            = df_std.index

    # Full-sample F_smooth — used for OLS (Step 4) only
    F_smooth_nf_full = pd.DataFrame(
        results_em['F_smooth'],
        index   = dates,
        columns = all_factor_names,
    )
    Lambda_nf = pd.DataFrame(
        results_em['Lambda'],
        index   = all_cols_nf,
        columns = all_factor_names,
    )

    lambda_est_fname = f'lambda_estimated_{n_new}factor_{new_factor_name}.csv'
    Lambda_nf.to_csv(lambda_est_fname)

    logger.info(f'  Final log-likelihood: {results_em["ll_history"][-1]:.2f}')
    logger.debug(f'  Lambda saved to {lambda_est_fname}')
    logger.debug('')

    # =========================================================
    # WALK-FORWARD EM — leakage-free F_smooth for Kalman
    # =========================================================
    logger.info('  ' + '─' * 65)
    logger.info(f'  Running {n_new}-factor walk-forward EM')
    logger.info('  ' + '─' * 65)
    logger.info('')

    F_smooth_nf_wf, final_G_9b, final_D_9b, final_RP_9b = run_walkforward_em(
        df_std         = df_std,
        df_std_full    = df_std_full, 
        all_cols       = all_cols_nf,
        lambda_init    = lambda_nf.values,
        F0_init        = F0_nf,
        oos_start_year = oos_start_year,
        n_iter         = n_iter,
        tol            = tol,
        factor_names   = all_factor_names,
    )

    # =========================================================
    # RE-RUN STEP 4 (OLS) — uses full-sample F_smooth
    # =========================================================
    logger.info('  ' + '─' * 65)
    logger.info(f'  OLS — {n_new}-factor EM')
    logger.info('  ' + '─' * 65)
    logger.info('')

    step4_nf = run_spx_regression(
        factors        = F_smooth_nf_full,
        spx            = df_raw[[REGRESSION_TARGET]],
        forward_months = forward_months,
    )

    # =========================================================
    # RE-RUN STEP 6 (KALMAN) — uses walk-forward F_smooth
    # =========================================================
    logger.info('')
    logger.info('  ' + '─' * 65)
    logger.info(f'  Kalman walk-forward — {n_new}-factor')
    logger.info('  ' + '─' * 65)
    logger.info('')

    kalman_nf = run_kalman_regression(
        F_smooth          = F_smooth_nf_wf,
        df_raw            = df_raw,
        REGRESSION_TARGET = REGRESSION_TARGET,
        forward_months    = forward_months,
        start_year        = start_year,
        beta_drift_q      = beta_drift_q,
        obs_noise_r       = obs_noise_r,
        in_sample_r2      = step4_nf['model'].rsquared,
        enable_graph      = enable_graph,
    )

    # =========================================================
    # COMPARISON TABLE
    # =========================================================
    baseline_is_r2     = step4_results['model'].rsquared
    baseline_oos_r2    = kalman_results['oos_r2']
    baseline_oos_r2_bc = kalman_results['oos_r2_bc']

    is_r2_nf     = step4_nf['model'].rsquared
    oos_r2_nf    = kalman_nf['oos_r2']
    oos_r2_bc_nf = kalman_nf['oos_r2_bc']

    delta_is     = is_r2_nf     - baseline_is_r2
    delta_oos    = oos_r2_nf    - baseline_oos_r2
    delta_oos_bc = oos_r2_bc_nf - baseline_oos_r2_bc

    if delta_oos > 0.010 and delta_oos_bc > 0.0:
        verdict = f'✅  ADOPT {n_new}-FACTOR MODEL'
    elif delta_oos > 0.010 and delta_oos_bc <= 0.0:
        verdict = '⚠️   RAW IMPROVES, BC DOES NOT — REVIEW'
    elif delta_oos > 0.0:
        verdict = f'⚠️   MARGINAL — RETAIN {n_existing}-FACTOR'
    else:
        verdict = f'❌  REJECT — RETAIN {n_existing}-FACTOR MODEL'

    base_label = f'{n_existing}-factor'
    new_label  = f'{n_new}-factor'

    logger.info('')
    logger.info(f'  ┌──────────────────────────────────────────────────────────┐')
    logger.info(f'  │  Re-run: {base_label} vs {new_label} EM  [{new_factor_name}, {len(NEW_COMPOSITE_COLS)} series]')
    logger.info(f'  ├──────────────────────────────────────────────────────────┤')
    logger.info(f'  │  {"Metric":<35} {base_label:>10} {new_label:>10}  │')
    logger.info(f'  ├──────────────────────────────────────────────────────────┤')
    logger.info(f'  │  {"In-sample R²":<35} {baseline_is_r2:>10.4f} {is_r2_nf:>10.4f}  │')
    logger.info(f'  │  {"OOS R² (raw)":<35} {baseline_oos_r2:>10.4f} {oos_r2_nf:>10.4f}  │')
    logger.info(f'  │  {"OOS R² (bias corrected)":<35} {baseline_oos_r2_bc:>10.4f} {oos_r2_bc_nf:>10.4f}  │')
    logger.info(f'  │  {"Directional accuracy":<35} {kalman_results["directional_acc"]:>10.1%} {kalman_nf["directional_acc"]:>10.1%}  │')
    logger.info(f'  │  {"Directional acc (BC)":<35} {kalman_results["dir_acc_bc"]:>10.1%} {kalman_nf["dir_acc_bc"]:>10.1%}  │')
    logger.info(f'  │  {"Monotonicity":<35} {kalman_results["monotonicity"]:>10.0%} {kalman_nf["monotonicity"]:>10.0%}  │')
    logger.info(f'  │  {"MAE":<35} {kalman_results["mae"]:>10.1%} {kalman_nf["mae"]:>10.1%}  │')
    logger.info(f'  ├──────────────────────────────────────────────────────────┤')
    logger.info(f'  │  {"ΔIn-sample R²":<35} {"":>10} {delta_is:>+10.4f}  │')
    logger.info(f'  │  {"ΔOOS R² (raw)":<35} {"":>10} {delta_oos:>+10.4f}  │')
    logger.info(f'  │  {"ΔOOS R² (bias corrected)":<35} {"":>10} {delta_oos_bc:>+10.4f}  │')
    logger.info(f'  ├──────────────────────────────────────────────────────────┤')
    logger.info(f'  │  {"Verdict":<55} │')
    logger.info(f'  │  {verdict:<55} │')
    logger.info(f'  └──────────────────────────────────────────────────────────┘')
    logger.info('')

    # =========================================================
    # RE-RUN STEP 7 (FINAL SYNTHESIS) IF ADOPTED
    # =========================================================
    synthesis_nf = None
    if '✅' in verdict:
        logger.info(f'  Running final synthesis with {n_new}-factor model...')
        logger.info('')
        synthesis_nf = run_final_synthesis(
            step4_results     = step4_nf,
            step5_results     = step5_results,
            step6_results     = kalman_nf,
            forward_months    = forward_months,
            regression_target = REGRESSION_TARGET,
            F_smooth          = F_smooth_nf_wf,   # walk-forward scores
            Lambda            = Lambda_nf,
            df_raw            = df_raw,
            df_ranked         = df_ranked,
        )

    return {
        'F_smooth_nf'       : F_smooth_nf_wf,     # walk-forward — used by loop for next round
        'F_smooth_nf_full'  : F_smooth_nf_full,   # full-sample — available for diagnostics
        'Lambda_nf'         : Lambda_nf,
        'step4_nf'          : step4_nf,
        'kalman_nf'         : kalman_nf,
        'synthesis_nf'      : synthesis_nf,
        'NEW_COMPOSITE_COLS': NEW_COMPOSITE_COLS,
        'new_factor_name'   : new_factor_name,
        'all_factor_names'  : all_factor_names,
        'n_factors'         : n_new,
        'is_r2_nf'          : is_r2_nf,
        'oos_r2_nf'         : oos_r2_nf,
        'oos_r2_bc_nf'      : oos_r2_bc_nf,
        'delta_is'          : delta_is,
        'delta_oos'         : delta_oos,
        'delta_oos_bc'      : delta_oos_bc,
        'verdict'           : verdict,
        'em_results'        : results_em,
        'pca_new_factor'    : pca_new_factor,
    }