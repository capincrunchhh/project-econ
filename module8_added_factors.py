import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
logger = logging.getLogger(__name__)


'''
Important note on potential data-leakage: 
---------------------------------------------------
As with module1_data_standardize.py, any there is potential data-leakage from manually assigning factors into buckets, along with PCA framework which contemplates
recent data. 

As with before, the fix, at least for PCA bucketing is to run PCA up until OOS start date.

THREE leakage points fixed in this version vs. prior:
  1. build_equal_weight_composite — sign alignment now fit on pre-OOS data only
  2. build_pca_composite          — scaler + PCA now fit on pre-OOS data only, then transform full sample
  3. test_composite (inside run_composite_factor_analysis) — R²/p-value scoring now uses pre-OOS data only

All three follow the same pattern:
  - FIT   on df_pre_oos (rows before oos_start_year)
  - APPLY to full df_std to produce composite scores over full history
  The composite series itself still covers the full date range —
  only the fitting/scoring is restricted to pre-OOS data.
'''


# =============================================================================
# COMPOSITE FACTOR DEFINITIONS
# Static dicts — exhaustive assignment of ALL series in the dataset.
# Every series belongs to exactly one composite regardless of whether it
# passed the R²/p-value filter or not.
#
# At runtime, the code filters each composite down to only the series
# currently in CURRENTLY_UNUSED_COLS before building the composite score.
# =============================================================================

COMPOSITE_CANDIDATES = {

    'Financial_Stress': [
        # Risk pricing, monetary conditions, funding stress — Layer 0/2/5
        'L0_treasury_10y',
        'L0_tips_10y',
        'L0_real_fed_funds',
        'L0_fed_balance_sheet',
        'L0_bank_reserves_yoy',
        'L0_bank_deposits_yoy',
        'L0_commercial_credit_yoy',
        'L0_ig_credit_spread',
        'L0_hy_credit_spread',
        'L0_lending_standards',
        'L0_financial_conditions',
        'L2_vix',
        'L2_nfci',
        'L2_commercial_paper_spread',
        'L2_mortgage_spread',
        'L2_credit_card_delinquency',
        'L2_dollar_index',
        'L2_m2_yoy',
        'L2_mortgage_debt_yoy',
        'L2_mortgage_share',
        'L2_liabilities_yoy',
        'L2_debt_service_ratio',
        'L2_pension_share',
        'L2_consumer_credit_dpi_ratio',
        'L2_corp_debt_gdp_ratio',
        'L2_corp_bond_debt_gdp_ratio',
        'L2_corp_bond_debt_yoy',
        'L2_corp_debt_yoy',
        'L2_corp_debt_profits_ratio',
        'L2_interest_profits_ratio',
        'L2_debt_dpi_ratio',
        'L4_breakeven_5y',
        'L4_breakeven_10y',
        'L4_forward_5y5y',
        'L5_equity_risk_premium',
        'L5_cape',
        'L5_sp500_pe',
    ],

    'Labor': [
        # Breadth of labor market and output utilization — Layer 0/1/4
        'L0_payroll_yoy',
        'L0_avg_weekly_hours',
        'L0_labor_force_participation',
        'L0_capacity_utilization',
        'L0_labor_productivity',
        'L0_total_factor_productivity',
        'L0_unit_labor_cost',
        'L1_ahe_yoy',
        'L1_eci_yoy',
        'L1_comp_per_hour_yoy',
        'L1_median_weekly_yoy',
        'L1_total_payroll_yoy',
        'L1_total_nonfarm_yoy',
        'L1_private_yoy',
        'L1_private_payrolls_yoy',
        'L1_household_employment_yoy',
        'L1_total_hours_worked_yoy',
        'L1_real_aggregate_payrolls_yoy',
        'L1_nominal_aggregate_payrolls_yoy',
        'L1_manufacturing_yoy',
        'L1_construction_yoy',
        'L1_mining_logging_yoy',
        'L1_information_yoy',
        'L1_financial_activities_yoy',
        'L1_professional_business_yoy',
        'L1_leisure_hospitality_yoy',
        'L1_trade_transport_utilities_yoy',
        'L1_other_services_yoy',
        'L1_government_yoy',
        'L1_education_health_yoy',
        'L4_indpro_yoy',
        'L4_mfg_production_yoy',
        'L4_mfg_shipments_yoy',
        'L4_capacity_util_total',
        'L4_capacity_util_mfg',
        'L1_mfg_new_orders_yoy',
        'L4_ulc_yoy',
    ],

    'Consumer': [
        # Household balance sheets, wealth, and spending decisions — Layer 2/3
        'L2_net_worth_yoy',
        'L2_net_worth_log',
        'L2_net_worth_dpi_ratio',
        'L2_financial_assets_networth_ratio',
        'L2_real_estate_networth_ratio',
        'L2_equity_exposure_ratio',
        'L2_liquid_assets_ratio',
        'L3_net_worth_dpi_ratio',
        'L3_personal_saving_rate',
        'L3_personal_saving_rate_yoy',
        'L3_spending_sustainability_ratio',
        'L3_consumer_credit_pce_ratio',
        'L3_consumer_credit_yoy',
        'L2_consumer_credit_yoy',
        'L3_pce_yoy',
        'L3_pce_services_yoy',
        'L3_pce_goods_yoy',
        'L3_durables_pce_share',
        'L3_retail_sales_yoy',
        'L3_core_retail_sales_yoy',
        'L4_retail_sales_yoy',
        'L1_retail_sales_yoy',
        'L3_real_dpi_yoy',
        'L1_real_disposable_income_yoy',
        'L1_disposable_income_yoy',
        'L1_transfers_dpi_ratio',
        'L4_michigan_5y',
    ],

    'Corporate': [
        # Profits, capex, margins, and energy as cost input — Layer 0/1/3/4
        'L1_corp_profits_aftertax_yoy',
        'L1_corp_profits_pretax_yoy',
        'L1_nonfinancial_profits_yoy',
        'L1_corp_cashflow_yoy',
        'L1_profit_margin',
        'L1_profit_margin_yoy',
        'L1_gdi_yoy',
        'L5_sp500_eps_yoy',
        'L4_corp_gva_yoy',
        'L4_corp_gva_gdp_ratio',
        'L4_nominal_gdp_yoy',
        'L4_real_final_sales_yoy',
        'L3_nonres_fixed_inv_yoy',
        'L3_equipment_inv_yoy',
        'L3_structures_inv_yoy',
        'L3_ip_inv_yoy',
        'L3_cap_goods_orders_yoy',
        'L0_nonres_fixed_investment_yoy',
        'L0_it_investment_yoy',
        'L3_inv_gdp_ratio',
        'L3_inv_profits_ratio',
        'L3_inventory_sales_ratio',
        'L3_inventory_change_gdp',
        'L3_retail_inventory_sales_ratio',
        'L0_inventory_consumption_ratio',
        'L0_inventory_change_rate',
        'L0_oil_price',
        'L0_gas_price',
        'L0_oil_volatility',
        'L0_crude_production',
        'L0_crude_inventories',
        'L0_energy_cpi',
        'L0_coal_share',
        'L0_natural_gas_share',
        'L0_nuclear_share',
        'L0_petroleum_share',
        'L0_renewables_share',
        'L4_ppi_yoy',
    ],

    'Government': [
        # Fiscal position, spending impulse, long-run expectations — Layer 3
        'L3_federal_expenditures_yoy',
        'L3_state_local_expenditures_yoy',
        'L3_real_govt_spending_yoy',
        'L3_defense_spending_yoy',
        'L3_federal_deficit_gdp_ratio',
        'L3_primary_deficit_gdp_ratio',
        'L3_interest_gdp_ratio',
        'L3_govt_spending_gdp_ratio',
        'L3_structures_share',
        'L1_social_benefits_yoy',
        'L1_current_transfers_yoy',
        'L1_unemployment_benefits_yoy',
        'L1_transfers_dpi_ratio',
        'L1_iva_yoy',
        'L4_gdp_yoy',
        'L4_output_gap',
        'L4_cpi_yoy',
        'L4_core_cpi_yoy',
        'L4_pce_yoy',
        'L4_core_pce_yoy',
        'L4_breakeven_10y',
        'L4_forward_5y5y',
        'L4_michigan_5y',
    ],
}


def build_equal_weight_composite(df_std, series_list, unused_set, name, oos_start_year=None):
    """
    Build an equal-weight composite factor from a list of series.

    Steps:
      1. Filter to series that are (a) in df_std and (b) in the current unused set
      2. Sign-align each series so all point in the same economic direction vs SPX
         (positive = bullish, negative = bearish) using the sign of their beta
         vs the standardized mean of available series as a proxy
      3. Average the sign-aligned standardized series → composite score

    --- LEAKAGE FIX vs. prior version ---
    Sign alignment betas are now estimated on pre-OOS data only (rows before
    oos_start_year). The resulting flip decisions are then applied to the full
    df_std to produce scores over the full history. Previously, betas were fit
    on the full sample including future data, which leaked post-OOS information
    into the sign alignment decision.

    Parameters
    ----------
    df_std         : pd.DataFrame — standardized data (all series, full history)
    series_list    : list         — candidate series for this composite
    unused_set     : set          — series currently unused by the 3-factor model
    name           : str          — composite name for logging
    oos_start_year : int|None     — if provided, sign alignment fit on pre-OOS data only

    Returns
    -------
    composite  : pd.Series    — equal-weight composite score, same index as df_std
    used_series: list         — series actually included after filtering
    """

    available = [s for s in series_list if s in df_std.columns and s in unused_set]

    if len(available) < 2:
        logger.debug(f'  {name} [EW]: only {len(available)} series available — skipping')
        return None, available

    df_sub = df_std[available].copy()

    # --- LEAKAGE FIX: subset to pre-OOS for sign alignment fitting ---
    # Prior version: used df_sub (full sample) to compute provisional_mean and betas
    # New version:   use df_pre_oos for fitting, then apply flip decisions to full df_sub
    if oos_start_year is not None:
        df_fit = df_sub.loc[df_sub.index < pd.Timestamp(f'{oos_start_year}-01-01')]
        logger.debug(f'  {name} [EW]: sign alignment fit on pre-OOS data ({len(df_fit)} rows)')
    else:
        df_fit = df_sub  # fallback: full sample (prior behavior)

    provisional_mean = df_fit.mean(axis=1)

    # Determine flip sign for each series using pre-OOS data only
    flip_signs = {}
    for col in df_sub.columns:
        pair = pd.concat([df_fit[col], provisional_mean], axis=1).dropna()
        if len(pair) < 20:
            flip_signs[col] = 1  # not enough pre-OOS data — no flip
            continue
        try:
            m = sm.OLS(pair.iloc[:, 1],
                       sm.add_constant(pair.iloc[:, 0])).fit()
            beta = m.params.iloc[1]
            flip_signs[col] = 1 if beta >= 0 else -1
        except Exception:
            flip_signs[col] = 1

    # Apply flip decisions to full sample (not just pre-OOS)
    aligned = pd.DataFrame(index=df_sub.index)
    for col in df_sub.columns:
        aligned[col] = df_sub[col] * flip_signs[col]

    composite = aligned.mean(axis=1)
    composite.name = f'{name}_EW'

    return composite, available


def build_pca_composite(df_std, series_list, unused_set, name, oos_start_year=None):
    """
    Build a PCA composite factor from a list of series.

    Extracts PC1 — the linear combination of series that explains the most
    variance across the group. Unlike equal-weight, series are weighted by
    their contribution to the dominant common signal, not equally.

    PC1 sign is oriented so that the dominant loading is positive
    (i.e. the factor moves in the direction most series move together).

    --- LEAKAGE FIX vs. prior version ---
    StandardScaler and PCA are now fit on pre-OOS data only (rows before
    oos_start_year). The fitted scaler and PCA are then used to transform
    the full sample, producing scores over the full history. Previously,
    both were fit on the full sample including future data, which leaked
    post-OOS variance structure into the PCA loadings and orientation.

    Parameters
    ----------
    df_std         : pd.DataFrame — standardized data (all series, full history)
    series_list    : list         — candidate series for this composite
    unused_set     : set          — series currently unused by the 3-factor model
    name           : str          — composite name for logging
    oos_start_year : int|None     — if provided, scaler + PCA fit on pre-OOS data only

    Returns
    -------
    composite       : pd.Series — PC1 composite score, same index as df_std
    used_series     : list      — series actually included after filtering
    explained_var   : float     — variance explained by PC1
    loadings        : pd.Series — PC1 loadings per series
    """

    available = [s for s in series_list if s in df_std.columns and s in unused_set]

    if len(available) < 2:
        logger.debug(f'  {name} [PCA]: only {len(available)} series available — skipping')
        return None, available, None, None

    df_sub = df_std[available].copy()

    threshold = int(len(available) * 0.33)  # composite start date = when at least 33% of series are non-NaN
    df_clean  = df_sub.dropna(thresh=threshold)
    df_clean  = df_clean.ffill().fillna(df_clean.mean())

    if len(df_clean) < 20:
        logger.debug(f'  {name} [PCA]: insufficient clean rows ({len(df_clean)}) — skipping')
        return None, available, None, None

    # --- LEAKAGE FIX: fit scaler and PCA on pre-OOS rows only ---
    # Prior version: fit on df_clean (full sample)
    # New version:   fit on df_pre_oos, then transform full df_clean
    if oos_start_year is not None:
        df_fit = df_clean.loc[df_clean.index < pd.Timestamp(f'{oos_start_year}-01-01')]
        if len(df_fit) < 20:
            # not enough pre-OOS data to fit PCA reliably — fall back to full sample
            logger.debug(f'  {name} [PCA]: insufficient pre-OOS rows ({len(df_fit)}) — falling back to full sample fit')
            df_fit = df_clean
        else:
            logger.debug(f'  {name} [PCA]: scaler + PCA fit on pre-OOS data ({len(df_fit)} rows)')
    else:
        df_fit = df_clean  # fallback: full sample (prior behavior)

    scaler   = StandardScaler()
    scaler.fit(df_fit)                          # FIT on pre-OOS only
    X_fit_scaled  = scaler.transform(df_fit)    # transform pre-OOS for PCA fitting
    X_full_scaled = scaler.transform(df_clean)  # transform full sample for scoring

    pca = PCA(n_components=1)
    pca.fit(X_fit_scaled)                       # FIT on pre-OOS only

    pc1 = pca.transform(X_full_scaled).flatten()  # TRANSFORM full sample

    explained_var = pca.explained_variance_ratio_[0]
    loadings      = pd.Series(pca.components_[0], index=available)

    if loadings.abs().idxmax() is not None:
        if loadings[loadings.abs().idxmax()] < 0:
            pc1      = -pc1
            loadings = -loadings

    composite = pd.Series(pc1, index=df_clean.index, name=f'{name}_PCA')
    composite = composite.reindex(df_std.index)

    return composite, available, explained_var, loadings


def run_composite_factor_analysis(
        df_std,
        df_raw,
        CURRENTLY_UNUSED_COLS,
        REGRESSION_TARGET,
        forward_months,
        F_smooth,
        oos_start_year=None,  # LEAKAGE FIX: new parameter — passed through to build functions and test_composite
):
    """
    Step 8 — Composite Factor Analysis

    Builds equal-weight and PCA versions of each composite candidate factor
    from the currently unused series pool. Displays side-by-side comparison
    of both versions including:
      - Series included (after filtering to unused)
      - Series dropped (in definition but already in model or missing)
      - Standalone R² and p-value vs SPX forward returns based on regression target + fwd months
      - Incremental R² vs existing 3-factor model
      - Current composite score
      - PCA variance explained and top loadings

    No walk-forward optimization at this stage — purely diagnostic.
    Walk-forward re-run happens in Step 9b once best composite is selected.

    --- LEAKAGE FIX vs. prior version ---
    oos_start_year is now passed to:
      1. build_equal_weight_composite — sign alignment fit on pre-OOS only
      2. build_pca_composite          — scaler + PCA fit on pre-OOS only
      3. test_composite (internal)    — R²/p-value scoring restricted to pre-OOS data
    Previously all three used the full sample, leaking post-OOS SPX returns
    into composite construction and candidate selection.

    Parameters
    ----------
    df_std               : pd.DataFrame — standardized data
    df_raw               : pd.DataFrame — raw unstandardized data
    CURRENTLY_UNUSED_COLS: list         — series unused by current 3-factor model
    REGRESSION_TARGET    : str          — e.g. 'L5_sp500_tr_6m'
    forward_months       : int          — prediction horizon
    F_smooth             : pd.DataFrame — current 3-factor Kalman-smoothed scores
    oos_start_year       : int|None     — if provided, all fitting restricted to pre-OOS data

    Returns
    -------
    results : dict — composite scores (EW and PCA) and diagnostics per candidate
    """

    unused_set = set(CURRENTLY_UNUSED_COLS)

    spx_forward = df_raw[[REGRESSION_TARGET]].copy()
    spx_forward['spx_fwd'] = spx_forward[REGRESSION_TARGET].shift(-forward_months)
    spx_forward = spx_forward[['spx_fwd']].dropna()

    # --- LEAKAGE FIX: restrict baseline R² regression to pre-OOS data ---
    # Prior version: baseline_model fit on full sample
    # New version:   fit on pre-OOS rows only so scoring reflects what was knowable at oos_start_year
    if oos_start_year is not None:
        spx_forward_fit = spx_forward.loc[spx_forward.index < pd.Timestamp(f'{oos_start_year}-01-01')]
        logger.debug(f'  Baseline R² regression fit on pre-OOS data ({len(spx_forward_fit)} rows)')
    else:
        spx_forward_fit = spx_forward  # fallback: full sample (prior behavior)

    baseline_X = sm.add_constant(
        F_smooth[['Growth', 'Discount', 'Risk_Premium']].reindex(spx_forward_fit.index)
    ).dropna()
    baseline_y     = spx_forward_fit['spx_fwd'].reindex(baseline_X.index).dropna()
    baseline_X     = baseline_X.reindex(baseline_y.index)
    baseline_model = sm.OLS(baseline_y, baseline_X).fit()
    baseline_r2    = baseline_model.rsquared

    # =========================================================
    # PRINT HEADER
    # =========================================================
    logger.info('=' * 65)
    logger.info('RECYCLING UNUSED SERIES INTO NEW COMPOSITE FACTORS')
    logger.info('=' * 65)
    logger.info(f'  Unused series available:    {len(unused_set)}')
    logger.info(f'  Composite candidates:       {len(COMPOSITE_CANDIDATES)}')
    logger.info(f'  Regression target:          {REGRESSION_TARGET}')
    logger.info(f'  Forward horizon:            {forward_months}m')
    logger.info(f'  3-factor in sample R²:       {baseline_r2:.4f}')
    if oos_start_year is not None:
        logger.info(f'  Leakage-free mode:          fitting restricted to pre-{oos_start_year} data')

    results = {}

    # =========================================================
    # COLUMN LAYOUT
    # Every data row: '  ' + metric(M chars, left) + col1(COL chars, right) + col2(COL chars, right)
    # =========================================================
    M   = 35   # metric label width
    COL = 18   # each value column width
    SEP = M + COL * 2

    def row(metric, v1='', v2=''):
        """Uniform data row — metric left-padded, both values right-padded to COL."""
        return f'  {metric:<{M}}{str(v1):>{COL}}{str(v2):>{COL}}'

    def fmt_f(v, signed=True):
        """Float → string. Returns '—' for None/NaN."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        return f'{v:+.4f}' if signed else f'{v:.4f}'

    def fmt_pct(v):
        """Float → percent string."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        return f'{v:.1%}'

    def fmt_pval(p):
        """p-value + significance stars as a single string (no trailing space)."""
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return '—'
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        return f'{p:.4f} {sig}'.strip()

    def get(stats, key):
        """Safe stats accessor — returns None on missing/NaN."""
        if stats is None:
            return None
        v = stats.get(key)
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    for name, series_list in COMPOSITE_CANDIDATES.items():

        logger.info('=' * 65)
        logger.info(f'  COMPOSITE: {name}')
        logger.info('=' * 65)

        in_unused     = [s for s in series_list if s in unused_set and s in df_std.columns]
        not_in_unused = [s for s in series_list if s not in unused_set]
        not_in_data   = [s for s in series_list if s not in df_std.columns]

        logger.info(f'  Defined:        {len(series_list)} series')
        logger.info(f'  Available:      {len(in_unused)} series (unused + in df_std)')
        logger.info(f'  In model:       {len(not_in_unused)} series (excluded — already active)')
        logger.info(f'  Missing data:   {len(not_in_data)} series (not in df_std)')
        logger.debug('')
        logger.debug(f'  Series included:')
        for s in in_unused:
            logger.debug(f'    {s}')
        if not_in_unused:
            logger.debug(f'  Series excluded (already in model):')
            for s in not_in_unused:
                logger.debug(f'    {s}')

        if len(in_unused) < 2:
            logger.debug(f'\n  Insufficient series — skipping {name}')
            continue

        # =========================================================
        # BUILD COMPOSITES
        # LEAKAGE FIX: oos_start_year now passed through to both builders
        # Prior version: oos_start_year not passed — full sample used internally
        # =========================================================
        ew_composite,  ew_series                      = build_equal_weight_composite(
            df_std, series_list, unused_set, name, oos_start_year=oos_start_year)
        pca_composite, pca_series, expl_var, loadings = build_pca_composite(
            df_std, series_list, unused_set, name, oos_start_year=oos_start_year)

        # =========================================================
        # REGRESSION TESTS
        # =========================================================
        def test_composite(composite, label):
            if composite is None:
                return None

            df_test = pd.concat([composite, spx_forward['spx_fwd']], axis=1).dropna()
            df_test.columns = ['composite', 'spx_fwd']

            # --- LEAKAGE FIX: restrict scoring regressions to pre-OOS data ---
            # Prior version: df_stride drawn from full sample including post-OOS SPX returns
            # New version:   df_stride restricted to pre-OOS rows for R²/p-value scoring
            #                The composite series itself still covers full history for downstream use
            if oos_start_year is not None:
                df_test_fit = df_test.loc[df_test.index < pd.Timestamp(f'{oos_start_year}-01-01')]
            else:
                df_test_fit = df_test  # fallback: full sample (prior behavior)

            df_stride = df_test_fit.iloc[::forward_months]

            if len(df_stride) < 20:
                logger.debug(f'  {label}: insufficient pre-OOS observations after stride ({len(df_stride)})')
                return None

            X_solo = sm.add_constant(df_stride[['composite']])
            y_solo = df_stride['spx_fwd']
            try:
                m_solo    = sm.OLS(y_solo, X_solo).fit()
                r2_solo   = m_solo.rsquared
                pval_solo = m_solo.pvalues['composite']
                beta_solo = m_solo.params['composite']
            except Exception:
                return None

            # Incremental R² vs existing factors — also restricted to pre-OOS
            factors_aligned = F_smooth[['Growth', 'Discount', 'Risk_Premium']].reindex(
                df_stride.index)
            df_incr = pd.concat(
                [factors_aligned, df_stride['composite'], df_stride['spx_fwd']],
                axis=1).dropna()

            X_incr = sm.add_constant(
                df_incr[['Growth', 'Discount', 'Risk_Premium', 'composite']])
            y_incr = df_incr['spx_fwd']
            try:
                m_incr    = sm.OLS(y_incr, X_incr).fit()
                r2_incr   = m_incr.rsquared
                pval_incr = m_incr.pvalues['composite']
                beta_incr = m_incr.params['composite']
                r2_delta  = r2_incr - baseline_r2
            except Exception:
                r2_incr = pval_incr = beta_incr = r2_delta = np.nan

            # Current score pulled from full composite (not pre-OOS subset)
            current_score = composite.dropna().iloc[-1]
            current_date  = composite.dropna().index[-1].strftime('%b %Y')

            return {
                'r2_solo'   : r2_solo,
                'pval_solo' : pval_solo,
                'beta_solo' : beta_solo,
                'r2_incr'   : r2_incr,
                'pval_incr' : pval_incr,
                'beta_incr' : beta_incr,
                'r2_delta'  : r2_delta,
                'current'   : current_score,
                'date'      : current_date,
                'n_obs'     : len(df_stride),
            }

        ew_stats  = test_composite(ew_composite,  f'{name} [EW]')
        pca_stats = test_composite(pca_composite, f'{name} [PCA]')

        # =========================================================
        # SIDE BY SIDE COMPARISON
        # All rows use row() so metric and value columns are always aligned.
        # =========================================================
        logger.debug('')
        logger.debug(f'  {"─" * SEP}')
        logger.debug(row('Metric', 'Equal Weight', 'PCA'))
        logger.debug(f'  {"─" * SEP}')

        logger.debug(row('Series used',
                  len(in_unused),
                  len(in_unused)))
        logger.debug(row('Observations (strided)',
                  str(get(ew_stats,  'n_obs') or '—'),
                  str(get(pca_stats, 'n_obs') or '—')))

        logger.debug('')
        logger.debug(f'  {"--- Standalone vs SPX ---"}')
        logger.debug(row('R² (solo)',
                  fmt_f(get(ew_stats,  'r2_solo'), signed=False),
                  fmt_f(get(pca_stats, 'r2_solo'), signed=False)))
        logger.debug(row('p-value (solo)',
                  fmt_pval(get(ew_stats,  'pval_solo')),
                  fmt_pval(get(pca_stats, 'pval_solo'))))
        logger.debug(row('Beta (solo)',
                  fmt_f(get(ew_stats,  'beta_solo')),
                  fmt_f(get(pca_stats, 'beta_solo'))))

        logger.debug('')
        logger.debug(f'  {"--- Incremental vs 3-factor model ---"}')
        logger.debug(row('Baseline in sample R² (3-factor)',
                  fmt_f(baseline_r2, signed=False),
                  fmt_f(baseline_r2, signed=False)))
        logger.debug(row('R² with composite (4-factor)',
                  fmt_f(get(ew_stats,  'r2_incr'), signed=False),
                  fmt_f(get(pca_stats, 'r2_incr'), signed=False)))
        logger.debug(row('ΔR² vs baseline',
                  fmt_f(get(ew_stats,  'r2_delta')),
                  fmt_f(get(pca_stats, 'r2_delta'))))
        logger.debug(row('p-value (incremental)',
                  fmt_pval(get(ew_stats,  'pval_incr')),
                  fmt_pval(get(pca_stats, 'pval_incr'))))

        logger.debug('')
        logger.debug(f'  {"--- Current reading ---"}')
        logger.debug(row('Composite score (current)',
                  fmt_f(get(ew_stats,  'current')),
                  fmt_f(get(pca_stats, 'current'))))
        logger.debug(row('As of',
                  ew_stats['date']  if ew_stats  else '—',
                  pca_stats['date'] if pca_stats else '—'))

        if pca_composite is not None and expl_var is not None:
            logger.debug('')
            logger.debug(f'  {"--- PCA diagnostics ---"}')
            logger.debug(row('Variance explained by PC1', fmt_pct(expl_var), ''))
            logger.debug(f'  Top 5 loadings:')
            top_loadings = loadings.abs().sort_values(ascending=False).head(5)
            for series in top_loadings.index:
                logger.debug(f'    {series:<40} {loadings[series]:>+.4f}')

        logger.debug(f'  {"─" * SEP}')

        results[name] = {
            'ew_composite'  : ew_composite,
            'pca_composite' : pca_composite,
            'ew_stats'      : ew_stats,
            'pca_stats'     : pca_stats,
            'series_used'   : in_unused,
            'expl_var'      : expl_var,
            'loadings'      : loadings,
        }

    # =========================================================
    # SUMMARY TABLE
    # =========================================================
    logger.info('=' * 65)
    logger.info('SUMMARY — ALL COMPOSITES')
    logger.info('=' * 65)
    logger.info(f'  {"Composite":<20} {"Method":>6} {"R²solo":>8} {"p-solo":>8} '
          f'{"R²incr":>8} {"ΔR²":>8} {"p-incr":>8} {"Current":>10} {"Pass?":>6}')
    logger.info(f'  {"─" * 65}')

    for name, res in results.items():
        for method, stats in [('EW', res['ew_stats']), ('PCA', res['pca_stats'])]:
            if stats is None:
                logger.info(f'  {name:<20} {method:>6} {"—":>8} {"—":>8} '
                      f'{"—":>8} {"—":>8} {"—":>8} {"—":>10} {"—":>6}')
                continue
            passes = (
                stats['r2_delta']  > 0.01 and
                stats['pval_incr'] < 0.05
            )
            flag = '✅' if passes else '❌'
            logger.info(f'  {name:<20} {method:>6}'
                  f' {stats["r2_solo"]:>8.4f}'
                  f' {stats["pval_solo"]:>8.4f}'
                  f' {stats["r2_incr"]:>8.4f}'
                  f' {stats["r2_delta"]:>+8.4f}'
                  f' {stats["pval_incr"]:>8.4f}'
                  f' {stats["current"]:>+10.4f}'
                  f' {flag:>6}')

    logger.info('')
    logger.info(f'  3-factor in sample R²: {baseline_r2:.4f}')
    logger.info(f'  Pass criteria: ΔR² > 0.01  AND  p-incr < 0.05')
    logger.info('')
    logger.info('  Next step: select best passing composite → re-run walk-forward')
    logger.info('  optimization to test OOS R² improvement.')

    return results