import pandas as pd
import time
from API_keys import EIA_api_key, FRED_api_key, BEA_api_key, BLS_api_key

from L0_all_data_get import L0_all_data_get
from L1_all_data_get import L1_all_data_get
from L2_all_data_get import L2_all_data_get
from L3_all_data_get import L3_all_data_get
from L4_all_data_get import L4_all_data_get
from L5_all_data_get import L5_all_data_get

'''
Important note on data leakage:
-------------------------------
A core requirement of any walk-forward validation framework is that the model
only sees data that would have been available in real-time at each point in time.
Raw FRED/BLS/BEA series are dated to their *reference period* (e.g. Q1 GDP is
dated January 1), but in practice that data is not published until weeks or
months later. Using undated raw series in a backtest therefore introduces
look-ahead bias -- the model implicitly "knows" data it could not have known.

apply_publication_lags() corrects this by shifting each series forward by its
empirical release lag before any downstream processing. A 3-month lag on GDP
means the January observation does not enter the dataset until April, matching
real-world availability. Lag assignments range from 0 months (real-time market
prices and spreads) to 12 months (BLS Total Factor Productivity, an annual
release). All constructed ratios inherit the lag of their slowest component.

This ensures the walk-forward OOS validation is a true real-time simulation:
at every training cutoff, the model is trained and evaluated exclusively on
data that a practitioner could have actually observed on that date.
'''


def prefix_columns(df, prefix):
    df.columns = [f"{prefix}{col}" for col in df.columns]
    return df


# -----------------------------------------------------------------------------
# Publication lag shift maps
# Each list contains column names sharing the same release lag.
#
# Lag categories:
#   0  months  -- real-time market/financial data (prices, spreads, indices)
#   1  month   -- monthly BLS/BEA releases (payrolls, CPI, retail sales, etc.)
#   2  months  -- slower monthly + quarterly BLS/BEA (PCE, bank data, ULC, productivity)
#   3  months  -- quarterly BEA/Fed Z.1 (GDP, profits, balance sheets, investment)
#   12 months  -- BLS annual releases (Total Factor Productivity)
# -----------------------------------------------------------------------------

LAG_0 = [
    # L0 -- market prices and financial conditions
    'L0_oil_price',
    'L0_gas_price',
    'L0_energy_cpi',
    'L0_oil_volatility',
    'L0_treasury_10y',
    'L0_tips_10y',
    'L0_real_fed_funds',
    'L0_ig_credit_spread',
    'L0_hy_credit_spread',
    'L0_fed_balance_sheet',
    'L0_financial_conditions',
    # L2 -- real-time financial conditions
    'L2_nfci',
    'L2_vix',
    'L2_dollar_index',
    'L2_commercial_paper_spread',
    # L4 -- market-implied inflation expectations (daily -> monthly)
    'L4_breakeven_5y',
    'L4_breakeven_10y',
    'L4_forward_5y5y',
    # L5 -- Shiller/Yale equity valuation (market prices)
    'L5_cape',
    'L5_sp500_log',
    'L5_sp500_tr_yoy',
    'L5_sp500_tr_6m',
    'L5_sp500_tr_3m',
    'L5_sp500_tr_1m',
    'L5_sp500_eps_yoy',
    'L5_sp500_pe',
    'L5_equity_risk_premium',
]

LAG_1 = [
    # L0 -- monthly BLS/Fed releases
    'L0_labor_force_participation',
    'L0_avg_weekly_hours',
    'L0_payroll_yoy',
    'L0_capacity_utilization',
    'L0_bank_deposits_yoy',
    'L0_commercial_credit_yoy',
    'L0_bank_reserves_yoy',
    # L1 -- BLS payrolls and wages (released ~4 weeks after reference month)
    'L1_total_payroll_yoy',
    'L1_total_payroll_log',
    'L1_private_payrolls_yoy',
    'L1_household_employment_yoy',
    'L1_nominal_aggregate_payrolls_yoy',
    'L1_total_nonfarm_yoy',
    'L1_private_yoy',
    'L1_mining_logging_yoy',
    'L1_construction_yoy',
    'L1_manufacturing_yoy',
    'L1_trade_transport_utilities_yoy',
    'L1_information_yoy',
    'L1_financial_activities_yoy',
    'L1_professional_business_yoy',
    'L1_education_health_yoy',
    'L1_leisure_hospitality_yoy',
    'L1_other_services_yoy',
    'L1_government_yoy',
    'L1_ahe_yoy',
    'L1_disposable_income_yoy',
    'L1_retail_sales_yoy',
    'L1_mfg_new_orders_yoy',
    # L2 -- monthly financial series
    'L2_m2_yoy',
    'L2_consumer_credit_yoy',
    # L3 -- monthly retail/orders/saving
    'L3_retail_sales_yoy',
    'L3_core_retail_sales_yoy',
    'L3_cap_goods_orders_yoy',
    'L3_inventory_sales_ratio',
    'L3_retail_inventory_sales_ratio',
    'L3_consumer_credit_yoy',
    'L3_personal_saving_rate',
    'L3_personal_saving_rate_yoy',
    # L4 -- monthly price/production releases
    'L4_cpi_yoy',
    'L4_core_cpi_yoy',
    'L4_ppi_yoy',
    'L4_indpro_yoy',
    'L4_mfg_production_yoy',
    'L4_capacity_util_total',
    'L4_capacity_util_mfg',
    'L4_michigan_5y',
    'L4_retail_sales_yoy',
    'L4_mfg_shipments_yoy',
]

LAG_2 = [
    # L0 -- EIA data (~2 month release lag) + quarterly BLS series
    'L0_crude_production',
    'L0_crude_inventories',
    'L0_inventory_change_rate',
    'L0_inventory_consumption_ratio',
    'L0_labor_productivity',        # BLS quarterly, ~2 quarter revision cycle
    'L0_unit_labor_cost',           # BLS quarterly, same
    'L0_lending_standards',         # Fed SLOOS, quarterly ~2 month lag
    # L1 -- BEA/BLS quarterly series with ~2 month effective lag
    'L1_real_aggregate_payrolls_yoy',
    'L1_total_hours_worked_yoy',
    'L1_real_disposable_income_yoy',
    'L1_social_benefits_yoy',
    'L1_current_transfers_yoy',
    'L1_unemployment_benefits_yoy',
    'L1_transfers_dpi_ratio',
    # L2 -- quarterly Fed/BEA series with ~2 month lag
    'L2_mortgage_debt_yoy',
    'L2_consumer_credit_dpi_ratio',
    'L2_credit_card_delinquency',
    'L2_debt_service_ratio',
    # L3 -- PCE and related (BEA, released ~2 months after reference month)
    'L3_pce_yoy',
    'L3_pce_services_yoy',
    'L3_pce_goods_yoy',
    'L3_spending_sustainability_ratio',
    'L3_consumer_credit_pce_ratio',
    'L3_durables_pce_share',
    'L3_real_dpi_yoy',
    # L4 -- PCE price indexes (BEA, ~2 month lag)
    'L4_pce_yoy',
    'L4_core_pce_yoy',
    'L4_ulc_yoy',                   # BLS quarterly ULC, same as L0_unit_labor_cost
]

LAG_3 = [
    # L0 -- quarterly BEA/BLS investment series
    'L0_nonres_fixed_investment_yoy',
    'L0_it_investment_yoy',
    # L1 -- BEA quarterly profit/income series (~3 month lag)
    'L1_corp_profits_aftertax_yoy',
    'L1_corp_profits_pretax_yoy',
    'L1_nonfinancial_profits_yoy',
    'L1_corp_cashflow_yoy',
    'L1_gdi_yoy',
    'L1_iva_yoy',
    'L1_profit_margin',
    'L1_profit_margin_yoy',
    'L1_comp_per_hour_yoy',
    'L1_median_weekly_yoy',
    'L1_eci_yoy',
    # L2 -- Fed Z.1 quarterly series (~3 month lag after quarter end)
    'L2_corp_debt_yoy',
    'L2_corp_bond_debt_yoy',
    'L2_corp_debt_gdp_ratio',
    'L2_corp_debt_profits_ratio',
    'L2_interest_profits_ratio',
    'L2_corp_bond_debt_gdp_ratio',
    'L2_net_worth_yoy',
    'L2_net_worth_log',
    'L2_net_worth_dpi_ratio',
    'L2_financial_assets_networth_ratio',
    'L2_real_estate_networth_ratio',
    'L2_equity_exposure_ratio',
    'L2_liquid_assets_ratio',
    'L2_pension_share',
    'L2_liabilities_yoy',
    'L2_mortgage_share',
    'L2_debt_dpi_ratio',
    # L3 -- BEA quarterly investment and government spending
    'L3_nonres_fixed_inv_yoy',
    'L3_equipment_inv_yoy',
    'L3_structures_inv_yoy',
    'L3_ip_inv_yoy',
    'L3_inv_gdp_ratio',
    'L3_inv_profits_ratio',
    'L3_structures_share',
    'L3_inventory_change_gdp',
    'L3_net_worth_dpi_ratio',
    'L3_real_govt_spending_yoy',
    'L3_federal_expenditures_yoy',
    'L3_state_local_expenditures_yoy',
    'L3_defense_spending_yoy',
    'L3_govt_spending_gdp_ratio',
    'L3_federal_deficit_gdp_ratio',
    'L3_primary_deficit_gdp_ratio',
    'L3_interest_gdp_ratio',
    # L4 -- BEA quarterly GDP and output series
    'L4_corp_gva_yoy',
    'L4_nominal_gdp_yoy',
    'L4_real_final_sales_yoy',
    'L4_corp_gva_gdp_ratio',
    'L4_gdp_yoy',
    'L4_output_gap',
]

LAG_12 = [
    # BLS annual releases -- MFP typically released in fall covering prior year
    'L0_total_factor_productivity',
]


def apply_publication_lags(df):
    """
    Shift each column forward by its publication lag so that the value
    for reference period T appears at T + lag. This ensures the pipeline
    only uses data that would have been available in real-time.

    A forward shift of N months means: the observation dated 'date' is
    moved to 'date + N months', reflecting when it was actually published.

    The index is temporarily extended by max_lag months to avoid shifted
    observations falling outside the index window and being silently dropped,
    then trimmed back to the original end date after all shifts are applied.
    """
    lag_map = (
        [(col,  0) for col in LAG_0]  +
        [(col,  1) for col in LAG_1]  +
        [(col,  2) for col in LAG_2]  +
        [(col,  3) for col in LAG_3]  +
        [(col, 12) for col in LAG_12]
    )

    # Warn on any columns in df not found in any lag bucket
    all_lagged = {col for col, _ in lag_map}
    unassigned = [col for col in df.columns if col not in all_lagged]
    if unassigned:
        print(f"WARNING: The following columns have no lag assignment and will NOT be shifted:\n  {unassigned}")

    original_end = df.index.max()
    max_lag      = 12

    # Extend index forward by max_lag months so shifted observations
    # are not silently dropped during reindex
    extended_index = pd.date_range(
        start=df.index.min(),
        end=original_end + pd.DateOffset(months=max_lag),
        freq='MS'
    )
    df = df.reindex(extended_index)

    for col, lag_months in lag_map:
        if col not in df.columns or lag_months == 0:
            continue
        # Extract non-NaN values, shift their index forward, reindex onto extended index
        original       = df[col].dropna().copy()
        original.index = original.index + pd.DateOffset(months=lag_months)
        df[col]        = original.reindex(df.index)

    # Trim back to original end date
    df = df.loc[:original_end]

    return df


def DFM_master_data_get(START_YEAR):
    print('MASTER: Starting Full DFM Data Collection...')

    L0_df = L0_all_data_get(EIA_api_key, FRED_api_key)
    L0_df = prefix_columns(L0_df, 'L0_')
    print('MASTER: Layer 0 Complete. Pausing before Layer 1...')
    time.sleep(45)

    L1_df = L1_all_data_get(FRED_api_key, BEA_api_key, BLS_api_key)
    L1_df = prefix_columns(L1_df, 'L1_')
    print('MASTER: Layer 1 Complete. Pausing before Layer 2...')
    time.sleep(45)

    L2_df = L2_all_data_get(FRED_api_key)
    L2_df = prefix_columns(L2_df, 'L2_')
    print('MASTER: Layer 2 Complete. Pausing before Layer 3...')
    time.sleep(45)

    L3_df = L3_all_data_get(FRED_api_key)
    L3_df = prefix_columns(L3_df, 'L3_')
    print('MASTER: Layer 3 Complete. Pausing before Layer 4...')
    time.sleep(45)

    L4_df = L4_all_data_get(FRED_api_key)
    L4_df = prefix_columns(L4_df, 'L4_')
    print('MASTER: Layer 4 Complete. Pausing before Layer 5...')
    time.sleep(45)

    L5_df = L5_all_data_get()
    L5_df = prefix_columns(L5_df, 'L5_')
    print('MASTER: Layer 5 Complete.')

    master_df = pd.concat([L0_df, L1_df, L2_df, L3_df, L4_df, L5_df], axis=1)
    master_df.index.name = 'date'

    # --- Apply publication lags ---
    print('MASTER: Applying publication lags...')
    master_df = apply_publication_lags(master_df)
    print('MASTER: Publication lags applied.')

    master_df = master_df.loc[f'{START_YEAR}-01-01':]

    print('MASTER: All Layer Data Collection Complete.')

    return master_df


#
# RUNNING THE CODE
#

#START_YEAR = 1990
#all_econ_data = DFM_master_data_get(START_YEAR)
#all_econ_data.to_csv('all_econ_data.csv')