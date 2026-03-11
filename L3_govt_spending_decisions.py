import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Government Spending Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Real spending flows (BEA NIPA via FRED, quarterly → monthly ffill):
# 1. GCEC1          real_govt_spending_yoy         real govt consumption & investment YoY      expressed as decimal. 5% = 0.05
# 2. FGEXPND        federal_expenditures_yoy       federal current expenditures YoY            expressed as decimal. 5% = 0.05  nominal billions
# 3. SLEXPND       state_local_expenditures_yoy   state & local current expenditures YoY       expressed as decimal. 5% = 0.05  nominal billions
# 4. FDEFX          defense_spending_yoy           federal defense consumption & invest YoY    expressed as decimal. 5% = 0.05  nominal billions

# Fiscal footprint ratio (level, quarterly → monthly ffill):
# 5. GCE / GDP      govt_spending_gdp_ratio        nominal govt spending / nominal GDP         expressed as ratio level
#    note: GCE (nominal) and GDP both in billions nominal → clean apples-to-apples ratio

# Fiscal sustainability ratios (level, quarterly → monthly ffill):
# 6. FGDEF / GDP             federal_deficit_gdp_ratio    net federal saving / GDP             expressed as ratio level
#    note: FGDEF billions nominal, GDP billions nominal → clean ratio
#    negative = deficit, positive = surplus. Quarterly BEA NIPA basis (vs OMB annual FYFSD)
# 7. (FGDEF+interest) / GDP  primary_deficit_gdp_ratio    primary balance / GDP                expressed as ratio level
#    note: primary balance strips out interest costs to show structural fiscal stance
#    all three series from BEA NIPA, quarterly, billions nominal → fully consistent
# 8. A091RC1Q027SBEA / GDP   interest_gdp_ratio           interest payments / GDP              expressed as ratio level
#    note: both billions nominal, both from BEA NIPA Table 3.2 / Table 1.1.5 → consistent


def get_government_spending(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Quarterly → dropna + forward-fill to monthly ---
    real_govt  = fred.get_series("GCEC1").dropna().resample("MS").ffill()
    gce        = fred.get_series("GCE").dropna().resample("MS").ffill()
    fed_expnd  = fred.get_series("FGEXPND").dropna().resample("MS").ffill()
    sl_expnd   = fred.get_series("SLEXPND").dropna().resample("MS").ffill()
    defense    = fred.get_series("FDEFX").dropna().resample("MS").ffill()
    gdp        = fred.get_series("GDP").dropna().resample("MS").ffill()
    interest   = fred.get_series("A091RC1Q027SBEA").dropna().resample("MS").ffill()
    fgdef      = fred.get_series("FGDEF").dropna().resample("MS").ffill()
    df = pd.DataFrame({
        "real_govt":   real_govt,
        "gce":         gce,
        "fed_expnd":   fed_expnd,
        "sl_expnd":    sl_expnd,
        "defense":     defense,
        "gdp":         gdp,
        "interest":    interest,
        "fgdef":       fgdef,
    })
    # --- YoY transformations ---
    df["real_govt_spending_yoy"]       = df["real_govt"].pct_change(12, fill_method=None)
    df["federal_expenditures_yoy"]     = df["fed_expnd"].pct_change(12, fill_method=None)
    df["state_local_expenditures_yoy"] = df["sl_expnd"].pct_change(12, fill_method=None)
    df["defense_spending_yoy"]         = df["defense"].pct_change(12, fill_method=None)
    # --- Fiscal footprint ratio ---
    df["govt_spending_gdp_ratio"]   = df["gce"]      / df["gdp"]
    # --- Fiscal sustainability ratios ---
    df["federal_deficit_gdp_ratio"] = df["fgdef"]                         / df["gdp"]
    df["primary_deficit_gdp_ratio"] = (df["fgdef"] + df["interest"])      / df["gdp"]
    df["interest_gdp_ratio"]        = df["interest"]                      / df["gdp"]
    # --- Drop raw levels ---
    df = df.drop(columns=["real_govt", "gce", "fed_expnd", "sl_expnd",
                           "defense", "gdp", "interest", "fgdef"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df


#
# RUNNING THE CODE:
#

#govt_spending = get_government_spending(FRED_api_key)
#print('L3: Running Government Spending Module. Gathering Data...')
#govt_spending.to_csv('government_spending.csv')
#print('L3: Government Spending Module Complete.')