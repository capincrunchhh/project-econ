import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Corporate Revenue Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Broad revenue proxies (BEA via FRED, quarterly → monthly ffill):
# 1. A449RC1Q027SBEA  corp_gva_yoy           corporate gross value added YoY                        expressed as decimal. 5% = 0.05  starts 1947
# 2. GDP              nominal_gdp_yoy        nominal GDP YoY                                        expressed as decimal. 5% = 0.05  starts 1947
# 3. FINSLC1          real_final_sales_yoy   real final sales domestic product YoY                  expressed as decimal. 5% = 0.05  starts 1947
#    note: FINSLC1 removes inventory swings → cleaner demand signal than GDP

# High-frequency revenue indicators (monthly):
# 4. RSAFS            retail_sales_yoy       nominal retail sales YoY                               expressed as decimal. 5% = 0.05  starts 1992
# 5. AMTMNO           mfg_shipments_yoy      manufacturers shipments YoY                            expressed as decimal. 5% = 0.05  starts 1992

# Derived ratios (level, quarterly → monthly ffill):
# 6. A449RC1Q027SBEA / GDP  corp_gva_gdp_ratio  corporate GVA / nominal GDP                         expressed as ratio level
#    note: A449RC1Q027SBEA in millions, GDP in billions → divide corp_gva by 1000 before ratio
#    structural signal for corporate sector share of economy


def get_corporate_revenue(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Quarterly → dropna + forward-fill to monthly ---
    corp_gva    = fred.get_series("A449RC1Q027SBEA").dropna().resample("MS").ffill()
    gdp         = fred.get_series("GDP").dropna().resample("MS").ffill()
    final_sales = fred.get_series("FINSLC1").dropna().resample("MS").ffill()
    # --- Monthly series ---
    retail_sales  = fred.get_series("RSAFS")
    mfg_shipments = fred.get_series("AMTMNO")
    df = pd.DataFrame({
        "corp_gva":       corp_gva,
        "gdp":            gdp,
        "final_sales":    final_sales,
        "retail_sales":   retail_sales,
        "mfg_shipments":  mfg_shipments,
    })
    # --- YoY transformations ---
    df["corp_gva_yoy"]         = df["corp_gva"].pct_change(12, fill_method=None)
    df["nominal_gdp_yoy"]      = df["gdp"].pct_change(12, fill_method=None)
    df["real_final_sales_yoy"] = df["final_sales"].pct_change(12, fill_method=None)
    df["retail_sales_yoy"]     = df["retail_sales"].pct_change(12, fill_method=None)
    df["mfg_shipments_yoy"]    = df["mfg_shipments"].pct_change(12, fill_method=None)
    # --- Derived ratio (level) ---
    df["corp_gva_gdp_ratio"] = (df["corp_gva"] / 1000) / df["gdp"] # divide by 1000 to normalize units
    # --- Drop raw levels ---
    df = df.drop(columns=["corp_gva", "gdp", "final_sales", "retail_sales", "mfg_shipments"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#corporate_revenue = get_corporate_revenue(FRED_api_key)
#print('L4: Running Corporate Revenue Module. Gathering Data...')
#corporate_revenue.to_csv('corporate_revenue.csv')
#print('L4: Corporate Revenue Module Complete.')