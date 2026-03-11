import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Corporate Leverage and Debt Variables. ALL DATA QUARTERLY → FORWARD-FILLED TO MONTHLY

# Corporate debt:
# 1. BCNSDODNS           corp_debt_yoy             nonfinancial corporate debt YoY              expressed as decimal. 5% = 0.05
# 2. derived             corp_debt_gdp_ratio        corporate debt / GDP                        expressed as ratio level
#    note: BCNSDODNS in millions, GDP in billions → divide BCNSDODNS by 1000
# 3. derived             corp_debt_profits_ratio    corporate debt / pretax profits             expressed as ratio level
#    note: BCNSDODNS in millions, CPATAX in billions → divide BCNSDODNS by 1000

# Interest burden:
# 4. derived             interest_profits_ratio     net interest payments / pretax profits      expressed as ratio level
#    note: A091RC1Q027SBEA in billions, CPATAX in billions → no unit adjustment needed

# Bond debt outstanding:
# 5. NCBDBIQ027S        corp_bond_debt_yoy         nonfinancial corp bond debt outstanding YoY  expressed as decimal. 5% = 0.05
# 6. derived            corp_bond_debt_gdp_ratio   corp bond debt outstanding / GDP             expressed as ratio level
#    note: NCBDBIQ027S in millions, GDP in billions → divide NCBDBIQ027S by 1000


def get_corporate_leverage(FRED_api_key):
    fred = Fred(FRED_api_key)

    # --- Quarterly series → dropna + forward-fill to monthly ---
    corp_debt     = fred.get_series("BCNSDODNS").dropna().resample("MS").ffill()
    gdp           = fred.get_series("GDP").dropna().resample("MS").ffill()
    cpatax        = fred.get_series("CPATAX").dropna().resample("MS").ffill()
    net_interest  = fred.get_series("A091RC1Q027SBEA").dropna().resample("MS").ffill()
    bond_issuance = fred.get_series("NCBDBIQ027S").dropna().resample("MS").ffill()

    df = pd.DataFrame({
        "corp_debt":     corp_debt,
        "gdp":           gdp,
        "cpatax":        cpatax,
        "net_interest":  net_interest,
        "bond_issuance": bond_issuance,
    })

    # --- YoY transformations ---
    df["corp_debt_yoy"]          = df["corp_debt"].pct_change(12, fill_method=None)
    df["corp_bond_debt_yoy"]     = df["bond_issuance"].pct_change(12, fill_method=None)

    # --- Derived ratios (level) ---
    # BCNSDODNS in millions, GDP in billions → divide by 1000
    df["corp_debt_gdp_ratio"]     = (df["corp_debt"]     / 1000) / df["gdp"]
    # BCNSDODNS in millions, CPATAX in billions → divide by 1000
    df["corp_debt_profits_ratio"] = (df["corp_debt"]     / 1000) / df["cpatax"]
    # A091RC1Q027SBEA in billions, CPATAX in billions → no unit adjustment
    df["interest_profits_ratio"]  =  df["net_interest"]           / df["cpatax"]
    # NCBDBIQ027S in millions, GDP in billions → divide by 1000
    df["corp_bond_debt_gdp_ratio"] = (df["bond_issuance"] / 1000) / df["gdp"]

    # --- Drop raw levels ---
    df = df.drop(columns=["corp_debt", "gdp", "cpatax", "net_interest", "bond_issuance"])

    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#corporate_leverage = get_corporate_leverage(FRED_api_key)
#print('L2: Running Corporate Leverage Module. Gathering Data...')
#corporate_leverage.to_csv('corporate_leverage.csv')
#print('L2: Corporate Leverage Module Complete.')