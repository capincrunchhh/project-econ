import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Inflation Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Consumer inflation (FRED, monthly):
# 1. CPIAUCSL    cpi_yoy        headline CPI YoY                                                expressed as decimal. 5% = 0.05  starts 1947
# 2. CPILFESL    core_cpi_yoy   core CPI (ex food & energy) YoY                                 expressed as decimal. 5% = 0.05  starts 1957
# 3. PCEPI       pce_yoy        headline PCE price index YoY                                    expressed as decimal. 5% = 0.05  starts 1959
# 4. PCEPILFE    core_pce_yoy   core PCE (ex food & energy) YoY                                 expressed as decimal. 5% = 0.05  starts 1959
#    note: core PCE is the Fed's preferred inflation gauge

# Producer / cost inflation (FRED):
# 5. PPIACO      ppi_yoy        producer price index YoY                                        expressed as decimal. 5% = 0.05  starts 1913  monthly
#    note: all-commodities PPI — long history back to 1913
#    modern final demand equivalent is PPIFIS (seasonally adjusted, starts Nov 2009)
#    no splice used — PPIACO provides consistent long-run pipeline cost signal
# 6. ULCNFB      ulc_yoy        unit labor cost YoY                                             expressed as decimal. 5% = 0.05  starts 1947  quarterly → monthly ffill

# Market-implied inflation expectations (FRED, daily → monthly average):
# 7. T5YIE      breakeven_5y       5-year breakeven inflation rate                              level (%)  starts 2003
# 8. T10YIE     breakeven_10y      10-year breakeven inflation rate                             level (%)  starts 2003
# 9. T5YIFR     forward_5y5y       5y5y forward inflation expectation                           level (%)  starts 2003
#    note: all daily series resampled to monthly mean

# Survey-based inflation expectations (FRED, monthly):
# 10. MICH      michigan_5y_exp    Michigan 5-year inflation expectations                       level (%)  starts 1978
#    note: kept as raw level — converting to YoY would destroy meaning
#    factor model can smooth; optional 12m rolling avg commented out below


def get_inflation(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Monthly series ---
    cpi      = fred.get_series("CPIAUCSL")
    core_cpi = fred.get_series("CPILFESL")
    pce      = fred.get_series("PCEPI")
    core_pce = fred.get_series("PCEPILFE")
    ppi      = fred.get_series("PPIACO")
    # --- Quarterly → dropna + forward-fill to monthly ---
    ulc = fred.get_series("ULCNFB").dropna().resample("MS").ffill()
    # --- Daily → monthly mean ---
    breakeven_5y  = fred.get_series("T5YIE").resample("MS").mean()
    breakeven_10y = fred.get_series("T10YIE").resample("MS").mean()
    forward_5y5y  = fred.get_series("T5YIFR").resample("MS").mean()
    # --- Monthly survey level ---
    michigan_5y = fred.get_series("MICH")
    df = pd.DataFrame({
        "cpi":           cpi,
        "core_cpi":      core_cpi,
        "pce":           pce,
        "core_pce":      core_pce,
        "ppi":           ppi,
        "ulc":           ulc,
        "breakeven_5y":  breakeven_5y,
        "breakeven_10y": breakeven_10y,
        "forward_5y5y":  forward_5y5y,
        "michigan_5y":   michigan_5y,
    })
    # --- YoY transformations (price index series only) ---
    df["cpi_yoy"]      = df["cpi"].pct_change(12, fill_method=None)
    df["core_cpi_yoy"] = df["core_cpi"].pct_change(12, fill_method=None)
    df["pce_yoy"]      = df["pce"].pct_change(12, fill_method=None)
    df["core_pce_yoy"] = df["core_pce"].pct_change(12, fill_method=None)
    df["ppi_yoy"]      = df["ppi"].pct_change(12, fill_method=None)
    df["ulc_yoy"]      = df["ulc"].pct_change(12, fill_method=None)
    # --- Optional: 12-month rolling average for Michigan (commented out — let model smooth) ---
    # df["michigan_5y_12m"] = df["michigan_5y"].rolling(12).mean()
    # --- Drop raw price index levels (keep market/survey levels as-is) ---
    df = df.drop(columns=["cpi", "core_cpi", "pce", "core_pce", "ppi", "ulc"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#inflation = get_inflation(FRED_api_key)
#print('L4: Running Inflation Module. Gathering Data...')
#inflation.to_csv('inflation.csv')
#print('L4: Inflation Module Complete.')