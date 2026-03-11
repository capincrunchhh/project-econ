import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
logger = logging.getLogger(__name__)


'''
Important note on potential data-leakage: 
---------------------------------------------------
The R²/p-value filter that decides which series go into Growth/Discount/Risk_Premium is run on the full sample including all future SPX returns. 
This means the factor composition was cherry-picked using information that wasn't available at the start of the OOS window. A series that happened 
to correlate with SPX over 2000-2026 gets included even if that correlation wasn't knowable in 1990. This can result in OOS R² inflating artificially.

The tricky part here is... if we manually assign series, we are baking in decades of historical evidence and human analyzed behavior to classify a series 
into one of the factors. that is implicit data-leakage in analog form. our systematic classification by regressing vs. anchor assets is similar, because 
the anchor assets reflect the aforementioned analog data leakage. and then we pile on that we are regressing over the full observed data-set, which is much 
of the same. 

it's extremely common and largely accepted in the academic and practitioner literature. Most published macro factor models use full-sample series selection. 
The justification is that the factor structure (what Growth, Discount, Risk_Premium mean economically) is based on decades of theoretical and empirical finance 
research — Fama/French, Campbell/Shiller, etc. The analog human knowledge you described is genuinely prior knowledge, not look-ahead. Classifying GDP as a growth 
indicator isn't leakage in any meaningful sense.
The systematic R²-based classification is slightly different — that's genuinely data-dependent. But even there, most practitioners would argue the correlations 
between GDP and equity returns, or credit spreads and risk premium, are stable structural relationships that would have been known at any point in the sample.

The fix therefore is to regress as-of the start date for the walk-forward optimization (start of OOS window) and classify into buckets that way.
--------------------------------------------------------------------------------------

Allocating to buckets this way may result in some NaN data i.e. starting at 1990 but data seires starts in 2002, being unclassified and moved to UNUSED
In module3_walkforward_em.py we call rank_and_assign_series as part of the walk-forward and dynamically update buckets over time as data becomes non-NaN

While this technically may reduce data leakage, the practical impact is likely small because:

The anchor relationships (GDP↔Growth, Treasury↔Discount, HY spread↔Risk_Premium) are stable over decades
Series that pass your R²/p-value filter using full-sample data would likely pass using 1970-1990 data too for the economically meaningful ones
The ones that wouldn't pass are probably spurious correlations anyway — so this change would actually improve model quality by removing them
'''


def rank_and_assign_series(df, target_col, forward_months, r2_threshold, pval_threshold, tiebreaker_gap, oos_start_year=None):
    """
    Ranks all input series by standalone R² vs forward SPX returns using
    non-overlapping annual observations to avoid autocorrelation.

    INCLUSION:      R² > r2_threshold AND p-value < pval_threshold vs forward SPX
    CLASSIFICATION: R² vs each reference anchor (GDP, Treasury, HY spread)
                    Assign to highest R² anchor
                    If gap between top two < tiebreaker_gap → use tiebreaker dict
                    If not in tiebreaker dict → UNUSED, flagged for review

    Three reference anchors are always included in their buckets regardless
    of R² filter. All other series are dynamic.

    If oos_start_year is provided, series selection and bucket assignment
    are performed using only data prior to that year — eliminating look-ahead
    bias in factor composition. Full df is still returned for downstream use.
    -------------------------------------------------------------------------
    Parameters
    ----------
    df              : pd.DataFrame — raw unstandardized data
    target_col      : str          — forward SPX column
    forward_months  : int          — forward shift for SPX target
    r2_threshold    : float        — minimum R² vs SPX to pass inclusion filter
    pval_threshold  : float        — maximum p-value vs SPX to pass inclusion filter
    tiebreaker_gap  : float        — minimum R² gap between top two anchors
                                     before falling back to tiebreaker dict
    oos_start_year  : int|None     — if provided, series ranking uses only data
                                     prior to this year (leakage-free bucket assignment)
    -------------------------------------------------------------------------
    Returns
    -------
    GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED, df_ranked
    """

    TARGET_VARIABLE = target_col

    # --- Subset df for ranking if oos_start_year provided ---
    if oos_start_year is not None:
        df_rank = df.loc[df.index < pd.Timestamp(f'{oos_start_year}-01-01')]
        logger.info(f'Series ranking using pre-OOS data only: {df_rank.index[0].strftime("%Y-%m")} → {df_rank.index[-1].strftime("%Y-%m")}  ({len(df_rank)} months)')
    else:
        df_rank = df
        logger.info('Series ranking using full sample data')

    # --- Three reference anchors ---
    ANCHOR_GROWTH       = 'L4_gdp_yoy'
    ANCHOR_DISCOUNT     = 'L0_treasury_10y'
    ANCHOR_RISK_PREMIUM = 'L0_hy_credit_spread'
    ALL_ANCHOR_COLS     = {ANCHOR_GROWTH, ANCHOR_DISCOUNT, ANCHOR_RISK_PREMIUM}

    # --- Always exclude regardless of R² ---
    ALWAYS_EXCLUDE = {
        TARGET_VARIABLE,
        'L5_sp500_tr_1m',
        'L5_sp500_tr_3m',
        'L5_sp500_tr_6m',
        'L5_sp500_tr_yoy',
    }

    # --- Tiebreaker dicts ---
    TIEBREAKER_GROWTH = {
        'L0_labor_force_participation', 'L0_avg_weekly_hours', 'L0_payroll_yoy',
        'L0_labor_productivity', 'L0_total_factor_productivity', 'L0_capacity_utilization',
        'L0_nonres_fixed_investment_yoy', 'L0_it_investment_yoy', 'L0_inventory_consumption_ratio',
        'L1_corp_profits_aftertax_yoy', 'L1_corp_profits_pretax_yoy', 'L1_nonfinancial_profits_yoy',
        'L1_corp_cashflow_yoy', 'L1_gdi_yoy', 'L1_mfg_new_orders_yoy', 'L1_retail_sales_yoy',
        'L1_profit_margin', 'L1_profit_margin_yoy', 'L1_total_payroll_yoy',
        'L1_real_aggregate_payrolls_yoy', 'L1_total_hours_worked_yoy', 'L1_private_payrolls_yoy',
        'L1_household_employment_yoy', 'L1_nominal_aggregate_payrolls_yoy', 'L1_total_nonfarm_yoy',
        'L1_private_yoy', 'L1_mining_logging_yoy', 'L1_construction_yoy', 'L1_manufacturing_yoy',
        'L1_trade_transport_utilities_yoy', 'L1_information_yoy', 'L1_financial_activities_yoy',
        'L1_professional_business_yoy', 'L1_education_health_yoy', 'L1_leisure_hospitality_yoy',
        'L1_other_services_yoy', 'L1_government_yoy', 'L1_disposable_income_yoy',
        'L1_real_disposable_income_yoy',
        'L3_nonres_fixed_inv_yoy', 'L3_equipment_inv_yoy', 'L3_structures_inv_yoy',
        'L3_ip_inv_yoy', 'L3_cap_goods_orders_yoy', 'L3_inv_gdp_ratio', 'L3_inv_profits_ratio',
        'L3_inventory_change_gdp', 'L3_inventory_sales_ratio', 'L3_pce_yoy',
        'L3_pce_services_yoy', 'L3_pce_goods_yoy', 'L3_retail_sales_yoy',
        'L3_core_retail_sales_yoy', 'L3_real_dpi_yoy', 'L3_durables_pce_share',
        'L3_real_govt_spending_yoy', 'L3_federal_expenditures_yoy',
        'L3_state_local_expenditures_yoy', 'L3_defense_spending_yoy',
        'L4_corp_gva_yoy', 'L4_nominal_gdp_yoy', 'L4_real_final_sales_yoy',
        'L4_retail_sales_yoy', 'L4_mfg_shipments_yoy', 'L4_corp_gva_gdp_ratio',
        'L4_capacity_util_total', 'L4_capacity_util_mfg', 'L4_output_gap',
        'L4_indpro_yoy', 'L4_mfg_production_yoy', 'L4_gdp_yoy',
        'L5_sp500_eps_yoy',
    }

    TIEBREAKER_DISCOUNT = {
        'L0_treasury_10y', 'L0_tips_10y', 'L0_real_fed_funds', 'L0_fed_balance_sheet',
        'L0_bank_deposits_yoy', 'L0_commercial_credit_yoy', 'L0_bank_reserves_yoy',
        'L0_coal_share', 'L0_natural_gas_share', 'L0_nuclear_share', 'L0_petroleum_share',
        'L0_renewables_share', 'L0_oil_price', 'L0_gas_price', 'L0_energy_cpi',
        'L0_crude_production', 'L0_crude_inventories', 'L0_inventory_change_rate',
        'L0_oil_volatility', 'L0_unit_labor_cost',
        'L1_iva_yoy', 'L1_social_benefits_yoy', 'L1_current_transfers_yoy',
        'L1_unemployment_benefits_yoy', 'L1_transfers_dpi_ratio', 'L1_ahe_yoy',
        'L1_eci_yoy', 'L1_comp_per_hour_yoy', 'L1_median_weekly_yoy',
        'L3_govt_spending_gdp_ratio', 'L3_federal_deficit_gdp_ratio',
        'L3_primary_deficit_gdp_ratio', 'L3_interest_gdp_ratio',
        'L3_personal_saving_rate', 'L3_personal_saving_rate_yoy',
        'L3_consumer_credit_pce_ratio', 'L3_net_worth_dpi_ratio',
        'L3_spending_sustainability_ratio', 'L3_retail_inventory_sales_ratio',
        'L3_structures_share', 'L3_consumer_credit_yoy',
        'L4_breakeven_5y', 'L4_breakeven_10y', 'L4_forward_5y5y', 'L4_michigan_5y',
        'L4_cpi_yoy', 'L4_core_cpi_yoy', 'L4_pce_yoy', 'L4_core_pce_yoy',
        'L4_ppi_yoy', 'L4_ulc_yoy',
    }

    TIEBREAKER_RISK_PREMIUM = {
        'L0_ig_credit_spread', 'L0_hy_credit_spread', 'L0_lending_standards',
        'L0_financial_conditions',
        'L2_corp_debt_yoy', 'L2_corp_bond_debt_yoy', 'L2_corp_debt_gdp_ratio',
        'L2_corp_debt_profits_ratio', 'L2_interest_profits_ratio', 'L2_corp_bond_debt_gdp_ratio',
        'L2_nfci', 'L2_vix', 'L2_commercial_paper_spread', 'L2_mortgage_spread',
        'L2_credit_card_delinquency', 'L2_liabilities_yoy', 'L2_mortgage_debt_yoy',
        'L2_consumer_credit_yoy', 'L2_debt_dpi_ratio', 'L2_mortgage_share',
        'L2_consumer_credit_dpi_ratio', 'L2_debt_service_ratio',
        'L2_m2_yoy', 'L2_dollar_index', 'L2_net_worth_yoy', 'L2_net_worth_log',
        'L2_net_worth_dpi_ratio', 'L2_financial_assets_networth_ratio',
        'L2_real_estate_networth_ratio', 'L2_equity_exposure_ratio',
        'L2_liquid_assets_ratio', 'L2_pension_share',
        'L5_cape', 'L5_sp500_pe', 'L5_equity_risk_premium',
    }

    TIEBREAKER_EXCLUDE = {
        'L1_total_payroll_log',
        'L5_sp500_log',
    }

    # --- Build forward-shifted SPX target using df_rank ---
    spx = df_rank[[target_col]].copy()
    spx['spx_forward'] = spx[target_col].shift(-forward_months)
    spx = spx[['spx_forward']].dropna()

    # --- Build anchor series for R²-based classification using df_rank ---
    anchors = df_rank[[ANCHOR_GROWTH, ANCHOR_DISCOUNT, ANCHOR_RISK_PREMIUM]].copy()

    # --- Columns to rank ---
    skip_cols = {target_col} | ALL_ANCHOR_COLS | ALWAYS_EXCLUDE | TIEBREAKER_EXCLUDE
    test_cols = [c for c in df_rank.columns if c not in skip_cols]

    results = []

    # track cols skipped due to insufficient history
    skipped_cols = []

    for col in test_cols:
        # --- Step 1: Inclusion — R² and p-value vs forward SPX ---
        spx_series = df_rank[[col]].join(spx, how='inner').dropna().iloc[::forward_months]
        if len(spx_series) < 10:  #require ~20 years of annual obs before promotion
            skipped_cols.append(col)  # <-- this line is the only addition needed
            continue

        try:
            model_spx = sm.OLS(
                spx_series['spx_forward'],
                sm.add_constant(spx_series[[col]])
            ).fit()
            r2_spx   = model_spx.rsquared
            beta_spx = model_spx.params[col]
            pval_spx = model_spx.pvalues[col]
            corr_spx = spx_series[col].corr(spx_series['spx_forward'])
            n        = len(spx_series)
        except Exception:
            continue

        passes_filter = r2_spx >= r2_threshold and pval_spx <= pval_threshold

        # --- Step 2: Classification — R² vs each anchor ---
        anchor_data = df_rank[[col]].join(anchors, how='inner').dropna()

        def r2_vs_anchor(anchor_col):
            pair = anchor_data[[col, anchor_col]].dropna()
            if len(pair) < 10:
                return 0.0
            try:
                m = sm.OLS(pair[anchor_col], sm.add_constant(pair[[col]])).fit()
                return m.rsquared
            except Exception:
                return 0.0

        r2_growth = r2_vs_anchor(ANCHOR_GROWTH)
        r2_disc   = r2_vs_anchor(ANCHOR_DISCOUNT)
        r2_rp     = r2_vs_anchor(ANCHOR_RISK_PREMIUM)

        anchor_r2s     = {'Growth': r2_growth, 'Discount': r2_disc, 'Risk_Premium': r2_rp}
        sorted_anchors = sorted(anchor_r2s.items(), key=lambda x: x[1], reverse=True)
        top_bucket,  top_r2    = sorted_anchors[0]
        _,           second_r2 = sorted_anchors[1]
        gap = top_r2 - second_r2

        # --- Step 3: Assign bucket ---
        if not passes_filter:
            assigned_bucket   = 'UNUSED'
            assignment_method = 'failed_filter'
        elif gap >= tiebreaker_gap:
            assigned_bucket   = top_bucket
            assignment_method = 'r2_with_factor_anchor'
        else:
            if col in TIEBREAKER_GROWTH:
                assigned_bucket   = 'Growth'
                assignment_method = 'tiebreaker'
            elif col in TIEBREAKER_DISCOUNT:
                assigned_bucket   = 'Discount'
                assignment_method = 'tiebreaker'
            elif col in TIEBREAKER_RISK_PREMIUM:
                assigned_bucket   = 'Risk_Premium'
                assignment_method = 'tiebreaker'
            else:
                assigned_bucket   = 'UNUSED'
                assignment_method = 'tiebreaker_missing'

        results.append({
            'series'            : col,
            'r2_spx'            : round(r2_spx,   4),
            'beta_spx'          : round(beta_spx,  4),
            'p_value'           : round(pval_spx,  4),
            'corr_spx'          : round(corr_spx,  3),
            'n_obs'             : n,
            'sig'               : '***' if pval_spx < 0.01 else '**' if pval_spx < 0.05 else '*' if pval_spx < 0.10 else '',
            'r2_growth'         : round(r2_growth, 4),
            'r2_discount'       : round(r2_disc,   4),
            'r2_riskprem'       : round(r2_rp,     4),
            'r2_gap'            : round(gap,        4),
            'assigned_bucket'   : assigned_bucket,
            'assignment_method' : assignment_method,
            'passes_filter'     : passes_filter,
        })

    df_ranked = pd.DataFrame(results).sort_values('r2_spx', ascending=False).reset_index(drop=True)

    # --- Build bucket lists starting from the three reference anchors only ---
    GROWTH_COLS       = [ANCHOR_GROWTH]
    DISCOUNT_COLS     = [ANCHOR_DISCOUNT]
    RISK_PREMIUM_COLS = [ANCHOR_RISK_PREMIUM]
    CURRENTLY_UNUSED  = list(ALWAYS_EXCLUDE | TIEBREAKER_EXCLUDE)
    # append skipped cols to CURRENTLY_UNUSED before returning
    CURRENTLY_UNUSED = list(dict.fromkeys(CURRENTLY_UNUSED + skipped_cols))
    if skipped_cols:
        logger.info(f'Skipped (insufficient pre-OOS history): {len(skipped_cols)} series → added to CURRENTLY_UNUSED')

    for _, row in df_ranked.iterrows():
        if row['assigned_bucket'] == 'Growth':
            GROWTH_COLS.append(row['series'])
        elif row['assigned_bucket'] == 'Discount':
            DISCOUNT_COLS.append(row['series'])
        elif row['assigned_bucket'] == 'Risk_Premium':
            RISK_PREMIUM_COLS.append(row['series'])
        else:
            CURRENTLY_UNUSED.append(row['series'])

    GROWTH_COLS       = list(dict.fromkeys(GROWTH_COLS))
    DISCOUNT_COLS     = list(dict.fromkeys(DISCOUNT_COLS))
    RISK_PREMIUM_COLS = list(dict.fromkeys(RISK_PREMIUM_COLS))
    CURRENTLY_UNUSED  = list(dict.fromkeys(CURRENTLY_UNUSED))

    # --- WARNING: tiebreaker_missing series ---
    tiebreaker_missing = df_ranked[df_ranked['assignment_method'] == 'tiebreaker_missing']
    if len(tiebreaker_missing) > 0:
        logger.warning(f'{len(tiebreaker_missing)} series passed filter but not in any tiebreaker dict:')
        for _, row in tiebreaker_missing.iterrows():
            logger.warning(
                f'  {row["series"]:<45}  r2_spx={row["r2_spx"]}  gap={row["r2_gap"]}  '
                f'r2_G={row["r2_growth"]}  r2_D={row["r2_discount"]}  r2_RP={row["r2_riskprem"]}'
            )
        logger.warning('  → Add these to a tiebreaker dict or TIEBREAKER_EXCLUDE')

    # --- INFO: summary counts ---
    n_pass = df_ranked['passes_filter'].sum()
    logger.info(f'Series ranking complete — tested: {len(df_ranked)}  passing filter: {n_pass}')
    logger.info(f'Inclusion: R² > {r2_threshold}  p-value < {pval_threshold}  tiebreaker gap: {tiebreaker_gap}')

    # --- DEBUG: full per-series R² table ---
    header = (
        f'  {"Series":<45} {"R²spx":>6} {"p-val":>8} {"R²G":>6} {"R²D":>6} '
        f'{"R²RP":>6} {"Gap":>6} {"Method":>12} {"Bucket":>12}'
    )
    logger.debug(f'Full series ranking vs {forward_months}m forward SPX:')
    logger.debug(header)
    logger.debug('  ' + '-' * 120)
    for _, row in df_ranked.iterrows():
        logger.debug(
            f'  {row["series"]:<45} {row["r2_spx"]:>6.4f} {row["p_value"]:>8.4f} '
            f'{row["r2_growth"]:>6.4f} {row["r2_discount"]:>6.4f} {row["r2_riskprem"]:>6.4f} '
            f'{row["r2_gap"]:>6.4f} {str(row["assignment_method"]):>12} '
            f'{str(row["assigned_bucket"]):>12}'
        )

    return GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED, df_ranked


def standardize_data(df, REGRESSION_TARGET, forward_months, r2_threshold, pval_threshold, tiebreaker_gap, oos_start_year=None):
    """
    Standardize input data and dynamically assign series to factor buckets.

    Produces two standardized DataFrames:
      - df_std    : full-sample z-scores         — used for Steps 2-5 (in-sample / diagnostic)
      - df_std_wf : expanding-window z-scores    — used for Steps 6+  (leakage-free)

    Runs rank_and_assign_series() twice:
      - Pre-OOS  (oos_start_year cutoff): leakage-free buckets for Steps 6+
      - Full sample                     : buckets for Steps 2-5

    Both rankings are validated and saved to CSV for inspection.
    """

    # --- Full-sample standardization (Steps 2-5) ---
    means  = df.mean(skipna=True)
    stds   = df.std(skipna=True)
    df_std = (df - means) / stds

    # --- Expanding-window standardization (Steps 6+) ---
    df_std_wf = df.copy().astype(float) * np.nan
    for col in df.columns:
        col_vals = df[col].values
        out      = np.full(len(df), np.nan)
        for t in range(1, len(df)):
            hist = df[col].iloc[:t].dropna()
            if len(hist) < 12:
                continue
            mu    = hist.mean()
            sigma = hist.std()
            if sigma > 0:
                out[t] = (col_vals[t] - mu) / sigma
        df_std_wf[col] = out

    TARGET_VARIABLE = REGRESSION_TARGET
    logger.info('=' * 65)

    # --- Pre-OOS ranking: leakage-free bucket assignment for Steps 6+ ---
    GROWTH_COLS_WF, DISCOUNT_COLS_WF, RISK_PREMIUM_COLS_WF, CURRENTLY_UNUSED_COLS_WF, df_ranked_wf = rank_and_assign_series(
        df             = df,
        target_col     = TARGET_VARIABLE,
        forward_months = forward_months,
        r2_threshold   = r2_threshold,
        pval_threshold = pval_threshold,
        tiebreaker_gap = tiebreaker_gap,
        oos_start_year = oos_start_year,
    )
    df_ranked_wf.to_csv('series_r2_ranking_presample.csv', index=False)

    # --- Full-sample ranking: bucket assignment for Steps 2-5 ---
    GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED_COLS, df_ranked = rank_and_assign_series(
        df             = df,
        target_col     = TARGET_VARIABLE,
        forward_months = forward_months,
        r2_threshold   = r2_threshold,
        pval_threshold = pval_threshold,
        tiebreaker_gap = tiebreaker_gap,
        oos_start_year = None,
    )
    df_ranked.to_csv('series_r2_ranking_fullsample.csv', index=False)

    # --- Validation helper ---
    def validate_buckets(growth, discount, rp, unused, df_ref, label):
        all_bucketed        = set(growth + discount + rp + unused)
        all_input_cols      = set(df_ref.columns) - {TARGET_VARIABLE}
        in_bucket_not_in_df = all_bucketed - set(df_ref.columns)
        in_df_not_in_bucket = all_input_cols - all_bucketed
        in_multiple_buckets = (
            set(growth)   & set(discount) |
            set(growth)   & set(rp)       |
            set(growth)   & set(unused)   |
            set(discount) & set(rp)       |
            set(discount) & set(unused)   |
            set(rp)       & set(unused)
        )
        logger.info(f'--- Bucket validation: {label} ---')
        logger.info(f'Growth bucket:       {len(growth)} series')
        logger.info(f'Discount bucket:     {len(discount)} series')
        logger.info(f'Risk Premium bucket: {len(rp)} series')
        logger.info(f'Currently unused:    {len(unused)} series')
        logger.info(f'Target:              {TARGET_VARIABLE}')
        logger.info(f'Total series:        {len(all_bucketed) + 1}')
        if in_bucket_not_in_df:
            logger.info(f'In bucket but not in dataframe: {in_bucket_not_in_df}')
        if in_df_not_in_bucket:
            logger.info(f'In dataframe but not bucketed:  {in_df_not_in_bucket}')
        if in_multiple_buckets:
            logger.info(f'Assigned to multiple buckets:   {in_multiple_buckets}')
        if not in_bucket_not_in_df and not in_df_not_in_bucket and not in_multiple_buckets:
            logger.info('Bucket assignment CLEAN — all series accounted for, no overlaps.')

    # --- Validate pre-OOS buckets (Steps 6+) ---
    validate_buckets(GROWTH_COLS_WF, DISCOUNT_COLS_WF, RISK_PREMIUM_COLS_WF, CURRENTLY_UNUSED_COLS_WF, df_std_wf, 'pre-OOS (Steps 6+)')

    # --- Validate full-sample buckets (Steps 2-5) ---
    validate_buckets(GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED_COLS, df_std, 'full-sample (Steps 2-5)')

    logger.info('Factor buckets assigned. Ready for PCA / EM.')

    return (
        df_std, df_std_wf,
        TARGET_VARIABLE,
        GROWTH_COLS,    DISCOUNT_COLS,    RISK_PREMIUM_COLS,    CURRENTLY_UNUSED_COLS,    df_ranked,
        GROWTH_COLS_WF, DISCOUNT_COLS_WF, RISK_PREMIUM_COLS_WF, CURRENTLY_UNUSED_COLS_WF, df_ranked_wf,
    )