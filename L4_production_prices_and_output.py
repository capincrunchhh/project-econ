import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Output & Production Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Industrial production (FRED, monthly):
# 1. INDPRO      indpro_yoy         industrial production index YoY                                             expressed as decimal. 5% = 0.05  starts 1919
# 2. IPMAN       mfg_production_yoy manufacturing production YoY                                                expressed as decimal. 5% = 0.05  starts 1919

# Capacity utilization (FRED, monthly):
# 3. TCU         capacity_util_total   total industry capacity utilization                                      level (%)  starts 1967
# 4. MCUMFN      capacity_util_mfg     manufacturing capacity utilization                                       level (%)  starts 1967
#    note: kept as levels — slack/overheating signal, YoY would lose economic meaning

# GDP & output gap (FRED, quarterly → monthly ffill):
# 5. GDPC1       gdp_yoy            real GDP YoY                                                                expressed as decimal. 5% = 0.05  starts 1947
# 6. GDPC1/GDPPOT output_gap        output gap (% deviation from potential)                                     level (%)  starts 1949
#    note: GDPPOT is CBO potential GDP, billions chained 2017 dollars — same units as GDPC1
#    output_gap = (GDPC1 - GDPPOT) / GDPPOT * 100
#    both quarterly, forward-filled to monthly together after gap calculation


def get_output_production(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Monthly series ---
    indpro = fred.get_series("INDPRO")
    ipman  = fred.get_series("IPMAN")
    tcu    = fred.get_series("TCU")
    mcumfn = fred.get_series("MCUMFN")
    # --- Quarterly GDP series ---
    gdpc1  = fred.get_series("GDPC1")
    gdppot = fred.get_series("GDPPOT")
    # --- Output gap: calculate at quarterly frequency before resampling ---
    gdp_df = pd.DataFrame({"gdpc1": gdpc1, "gdppot": gdppot}).dropna()
    gdp_df["output_gap"] = (gdp_df["gdpc1"] - gdp_df["gdppot"]) / gdp_df["gdppot"] * 100
    # --- Quarterly → monthly ffill ---
    gdp_yoy_monthly    = gdp_df["gdpc1"].resample("MS").ffill()
    output_gap_monthly = gdp_df["output_gap"].resample("MS").ffill()
    df = pd.DataFrame({
        "indpro":               indpro,
        "mfg_production":       ipman,
        "capacity_util_total":  tcu,
        "capacity_util_mfg":    mcumfn,
        "gdp":                  gdp_yoy_monthly,
        "output_gap":           output_gap_monthly,
    })
    # --- YoY transformations ---
    df["indpro_yoy"]         = df["indpro"].pct_change(12, fill_method=None)
    df["mfg_production_yoy"] = df["mfg_production"].pct_change(12, fill_method=None)
    df["gdp_yoy"]            = df["gdp"].pct_change(12, fill_method=None)
    # --- Drop raw levels (keep capacity util and output gap as levels) ---
    df = df.drop(columns=["indpro", "mfg_production", "gdp"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#output_production = get_output_production(FRED_api_key)
#print('L4: Running Output & Production Module. Gathering Data...')
#output_production.to_csv('output_production.csv')
#print('L4: Inflation Module Complete.')