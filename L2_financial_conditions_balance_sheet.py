import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Financial Conditions Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Composite conditions:
# 1. NFCI              nfci                    Chicago Fed national financial conditions index      level (negative = accommodative)
#    note: weekly → monthly average

# Risk aversion:
# 2. VIXCLS            vix                     CBOE VIX equity volatility index                     level (index value)
#    note: daily → monthly average

# Funding stress:
# 3. CPN3M - TB3MS     commercial_paper_spread commercial paper spread over T-bill                  expressed as 1.05 = 1.05%
#    note: both monthly, spread = CPN3M minus TB3MS

# Dollar liquidity:
# 4. DTWEXBGS          dollar_index            trade-weighted US dollar index (broad)               level (index value)
#    note: daily → monthly average

# Housing credit stress:
# 5. MORTGAGE30US - DGS10   mortgage_spread    30Y mortgage rate minus 10Y treasury                 expressed as 1.05 = 1.05%
#    note: MORTGAGE30US weekly → monthly average, DGS10 daily → monthly average

# Money supply:
# 6. M2SL              m2_yoy                  M2 money supply YoY                                  expressed as decimal. 5% = 0.05
#    note: monthly

# Consumer credit stress:
# 7. DRCCLACBS         credit_card_delinquency credit card delinquency rate                         expressed as 1.05 = 1.05%
#    note: quarterly → forward-fill to monthly


def get_financial_conditions(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Weekly → monthly average ---
    nfci     = fred.get_series("NFCI").resample("MS").mean()
    # --- Daily → monthly average ---
    vix      = fred.get_series("VIXCLS").resample("MS").mean()
    # --- spliced dollar ---
    dollar_old = fred.get_series("DTWEXM").resample("MS").mean()      # 1973-2006
    dollar_new = fred.get_series("DTWEXBGS").resample("MS").mean()    # 2006-present
    dollar = pd.concat([dollar_old, dollar_new]).sort_index()
    dollar = dollar[~dollar.index.duplicated(keep="last")]
    dgs10    = fred.get_series("DGS10").resample("MS").mean()
    # --- Weekly → monthly average ---
    mortgage = fred.get_series("MORTGAGE30US").resample("MS").mean()
    # --- Monthly series ---
    cpn3m    = fred.get_series("CPN3M")
    tb3ms    = fred.get_series("TB3MS")
    m2       = fred.get_series("M2SL")
    # --- Quarterly → dropna + forward-fill to monthly ---
    cc_delinquency = fred.get_series("DRCCLACBS").dropna().resample("MS").ffill()
    df = pd.DataFrame({
        "nfci":            nfci,
        "vix":             vix,
        "dollar":          dollar,
        "dgs10":           dgs10,
        "mortgage":        mortgage,
        "cpn3m":           cpn3m,
        "tb3ms":           tb3ms,
        "m2":              m2,
        "cc_delinquency":  cc_delinquency,
    })
    # --- Spreads (level, expressed as 1.05 = 1.05%) ---
    df["commercial_paper_spread"] = df["cpn3m"]    - df["tb3ms"]
    df["mortgage_spread"]         = df["mortgage"] - df["dgs10"]
    # --- YoY ---
    df["m2_yoy"] = df["m2"].pct_change(12, fill_method=None)
    # --- Rename levels ---
    df["nfci"]                  = df["nfci"]
    df["vix"]                   = df["vix"]
    df["dollar_index"]          = df["dollar"]
    df["credit_card_delinquency"] = df["cc_delinquency"]
    # --- Drop raws ---
    df = df.drop(columns=["dollar", "dgs10", "mortgage", "cpn3m", "tb3ms", "m2", "cc_delinquency"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#financial_conditions = get_financial_conditions(FRED_api_key)
#print('L2: Running Financial Conditions Module. Gathering Data...')
#financial_conditions.to_csv('financial_conditions.csv')
#print('L2: Financial Conditions Module Complete.')