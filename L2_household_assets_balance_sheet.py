import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Household Balance Sheet Variables. ALL DATA QUARTERLY → FORWARD-FILLED TO MONTHLY

# Net worth:
# 1. BOGZ1FL192090005Q   net_worth_yoy          household net worth YoY               expressed as decimal. 5% = 0.05
# 2. BOGZ1FL192090005Q   net_worth_log          household net worth log level         expressed as log(millions USD)
# 3. derived             net_worth_dpi_ratio    net worth / DPI                       expressed as ratio level

# Asset composition ratios (all derived, level):
# 4. BOGZ1FL194090005Q / BOGZ1FL192090005Q   financial_assets_networth_ratio          financial assets / net worth                                  ratio
# 5. OEHRENWBSHNO       / BOGZ1FL192090005Q   real_estate_networth_ratio              real estate equity / net worth                                ratio
# 6. BOGZ1LM193064005Q  / BOGZ1FL194090005Q   equity_exposure_ratio                   equities + mutual funds / financial assets                    ratio
# 7. (BOGZ1FL193020005Q + BOGZ1FL193034005Q) / BOGZ1FL194090005Q                      liquid_assets_ratio   (deposits + MMF) / financial assets     ratio
# 8. HNOPFAQ027S        / BOGZ1FL194090005Q   pension_share                           pension entitlements / financial assets                       ratio


def get_household_balance_sheet(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Pull all quarterly series → forward-fill to monthly ---
    net_worth        = fred.get_series("BOGZ1FL192090005Q").dropna().resample("MS").ffill()
    financial_assets = fred.get_series("BOGZ1FL194090005Q").dropna().resample("MS").ffill()
    real_estate      = fred.get_series("OEHRENWBSHNO").dropna().resample("MS").ffill()
    equities_mf      = fred.get_series("BOGZ1LM193064005Q").dropna().resample("MS").ffill()
    deposits         = fred.get_series("BOGZ1FL193020005Q").dropna().resample("MS").ffill()
    mmf              = fred.get_series("BOGZ1FL193034005Q").dropna().resample("MS").ffill()
    pension          = fred.get_series("HNOPFAQ027S").dropna().resample("MS").ffill()
    dpi              = fred.get_series("DPI").dropna().resample("MS").ffill()
    df = pd.DataFrame({
        "net_worth":        net_worth,
        "financial_assets": financial_assets,
        "real_estate":      real_estate,
        "equities_mf":      equities_mf,
        "deposits":         deposits,
        "mmf":              mmf,
        "pension":          pension,
        "dpi":              dpi,
    })
    # --- YoY and log transformations ---
    df["net_worth_yoy"] = df["net_worth"].pct_change(12, fill_method=None)
    df["net_worth_log"] = np.log(df["net_worth"])
    # --- Derived ratios (level) ---
    df["net_worth_dpi_ratio"]             = (df["net_worth"] / 1000) / df["dpi"] #net worth in millions, DPI in billions
    df["financial_assets_networth_ratio"] = df["financial_assets"] / df["net_worth"]
    df["real_estate_networth_ratio"]      = df["real_estate"]      / df["net_worth"]
    df["equity_exposure_ratio"]           = df["equities_mf"]      / df["financial_assets"]
    df["liquid_assets_ratio"]             = (df["deposits"] + df["mmf"]) / df["financial_assets"]
    df["pension_share"]                   = df["pension"]          / df["financial_assets"]
    # --- Drop raw levels ---
    df = df.drop(columns=["net_worth", "financial_assets", "real_estate",
                           "equities_mf", "deposits", "mmf", "pension", "dpi"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#household_balance_sheet = get_household_balance_sheet(FRED_api_key)
#print('L2: Running Household Balance Sheet Module. Gathering Data...')
#household_balance_sheet.to_csv('household_balance_sheet.csv')
#print('L2: Household Balance Sheet Module Complete.')