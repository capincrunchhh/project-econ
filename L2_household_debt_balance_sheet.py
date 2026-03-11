import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Household Debt and Liabilities Variables. ALL DATA QUARTERLY → FORWARD-FILLED TO MONTHLY

# Total liabilities:
# 1. BOGZ1FL194190005Q  liabilities_yoy          total household liabilities YoY        expressed as decimal. 5% = 0.05
# 2. derived            debt_dpi_ratio            total liabilities / DPI               expressed as ratio level
#    note: liabilities in millions, DPI in billions → divide liabilities by 1000

# Debt service:
# 3. TDSP               debt_service_ratio        household debt service ratio          expressed as % of disposable income
#    note: starts 2005Q1, no pre-2005 equivalent available on FRED

# Mortgage debt:
# 4. HHMSDODNS          mortgage_debt_yoy         household mortgage debt YoY           expressed as decimal. 5% = 0.05
# 5. derived            mortgage_share            mortgage debt / total liabilities     expressed as ratio level

# Consumer credit:
# 6. TOTALSL            consumer_credit_yoy       total consumer credit YoY             expressed as decimal. 5% = 0.05
# 7. derived            consumer_credit_dpi_ratio consumer credit / DPI                 expressed as ratio level
#    note: TOTALSL in millions, DPI in billions → divide TOTALSL by 1000


def get_household_debt(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Quarterly series → dropna + forward-fill to monthly ---
    liabilities   = fred.get_series("BOGZ1FL194190005Q").dropna().resample("MS").ffill()
    mortgage_debt = fred.get_series("HHMSDODNS").dropna().resample("MS").ffill()
    debt_service  = fred.get_series("TDSP").dropna().resample("MS").ffill()
    dpi           = fred.get_series("DPI").dropna().resample("MS").ffill()
    # --- Monthly series → dropna + resample to MS ---
    consumer_credit = fred.get_series("TOTALSL").dropna().resample("MS").ffill()
    df = pd.DataFrame({
        "liabilities":      liabilities,
        "mortgage_debt":    mortgage_debt,
        "debt_service":     debt_service,
        "consumer_credit":  consumer_credit,
        "dpi":              dpi,
    })
    # --- YoY transformations ---
    df["liabilities_yoy"]     = df["liabilities"].pct_change(12, fill_method=None)
    df["mortgage_debt_yoy"]   = df["mortgage_debt"].pct_change(12, fill_method=None)
    df["consumer_credit_yoy"] = df["consumer_credit"].pct_change(12, fill_method=None)
    # --- Derived ratios (level) ---
    # liabilities in millions, DPI in billions → divide by 1000
    df["debt_dpi_ratio"]            = (df["liabilities"]    / 1000) / df["dpi"]
    df["mortgage_share"]            = df["mortgage_debt"]            / df["liabilities"]
    # TOTALSL in millions, DPI in billions → divide by 1000
    df["consumer_credit_dpi_ratio"] = (df["consumer_credit"] / 1000) / df["dpi"]
    # --- Debt service kept as level ---
    df["debt_service_ratio"] = df["debt_service"]
    # --- Drop raw levels ---
    df = df.drop(columns=["liabilities", "mortgage_debt", "debt_service",
                           "consumer_credit", "dpi"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#household_debt = get_household_debt(FRED_api_key)
#print('L2: Running Household Debt Module. Gathering Data...')
#household_debt.to_csv('household_debt.csv')
#print('L2: Household Debt Module Complete.')