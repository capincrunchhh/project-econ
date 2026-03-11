import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Corporate Investment & Capex Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Fixed investment flows (BEA via FRED, quarterly → monthly ffill):
# 1. PNFIC1               nonres_fixed_inv_yoy        real nonresidential fixed investment YoY       expressed as decimal. 5% = 0.05
# 2. Y033RX1Q020SBEA      equipment_inv_yoy           real equipment investment YoY                  expressed as decimal. 5% = 0.05  starts 2007
# 3. B009RX1Q020SBEA      structures_inv_yoy          real structures investment YoY                 expressed as decimal. 5% = 0.05  starts 2007
# 4. Y001RX1Q020SBEA      ip_inv_yoy                  real intellectual property investment YoY      expressed as decimal. 5% = 0.05  starts 2007

# Leading indicators (monthly):
# 5. DGORDER              cap_goods_orders_yoy        core capital goods orders (nondefense ex-air)  expressed as decimal. 5% = 0.05

# Derived ratios (level, quarterly → monthly ffill):
# 6. PNFI / GDP           inv_gdp_ratio               nominal investment / nominal GDP               expressed as ratio level
# 7. PNFI / CPATAX        inv_profits_ratio           nominal investment / after-tax corporate       expressed as ratio level
#    note: PNFI and CPATAX both in billions nominal → no unit adjustment
# 8. B009RX1Q020SBEA /    structures_share            real structures / real total investment        expressed as ratio level
#    PNFIC1                                           note: both real chained → consistent

# Inventory variables (quarterly → monthly ffill):
# 9. CBI / GDP           inventory_change_gdp        change in private inventories / GDP            expressed as ratio level
#     note: CBI in billions nominal, GDP in billions nominal → no unit adjustment
# 10. ISRATIO             inventory_sales_ratio       total business inventory-to-sales ratio        expressed as level


def get_corporate_investment(FRED_api_key):
    fred = Fred(FRED_api_key)

    # --- Quarterly series → dropna + forward-fill to monthly ---
    nonres_fixed_inv = fred.get_series("PNFIC1").dropna().resample("MS").ffill()
    equipment_inv    = fred.get_series("Y033RX1Q020SBEA").dropna().resample("MS").ffill()
    structures_inv   = fred.get_series("B009RX1Q020SBEA").dropna().resample("MS").ffill()
    ip_inv           = fred.get_series("Y001RX1Q020SBEA").dropna().resample("MS").ffill()
    # Nominal series for ratios (both billions nominal → consistent)
    pnfi             = fred.get_series("PNFI").dropna().resample("MS").ffill()
    gdp              = fred.get_series("GDP").dropna().resample("MS").ffill()
    cpatax           = fred.get_series("CPATAX").dropna().resample("MS").ffill()
    cbi              = fred.get_series("CBI").dropna().resample("MS").ffill()
    # --- Monthly series ---
    cap_goods_orders = fred.get_series("DGORDER")
    isratio          = fred.get_series("ISRATIO")
    df = pd.DataFrame({
        "nonres_fixed_inv":  nonres_fixed_inv,
        "equipment_inv":     equipment_inv,
        "structures_inv":    structures_inv,
        "ip_inv":            ip_inv,
        "pnfi":              pnfi,
        "gdp":               gdp,
        "cpatax":            cpatax,
        "cbi":               cbi,
        "cap_goods_orders":  cap_goods_orders,
        "isratio":           isratio,
    })
    # --- YoY transformations ---
    df["nonres_fixed_inv_yoy"]  = df["nonres_fixed_inv"].pct_change(12, fill_method=None)
    df["equipment_inv_yoy"]     = df["equipment_inv"].pct_change(12, fill_method=None)
    df["structures_inv_yoy"]    = df["structures_inv"].pct_change(12, fill_method=None)
    df["ip_inv_yoy"]            = df["ip_inv"].pct_change(12, fill_method=None)
    df["cap_goods_orders_yoy"]  = df["cap_goods_orders"].pct_change(12, fill_method=None)
    # --- Derived ratios (level) ---
    df["inv_gdp_ratio"]         = df["pnfi"]           / df["gdp"]
    df["inv_profits_ratio"]     = df["pnfi"]           / df["cpatax"]
    df["structures_share"]      = df["structures_inv"] / df["nonres_fixed_inv"]
    df["inventory_change_gdp"]  = df["cbi"]            / df["gdp"]
    df["inventory_sales_ratio"] = df["isratio"]
    # --- Drop raw levels ---
    df = df.drop(columns=["nonres_fixed_inv", "equipment_inv", "structures_inv",
                           "ip_inv", "pnfi", "gdp", "cpatax", "cbi", "cap_goods_orders", "isratio"])
    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#corp_investment = get_corporate_investment(FRED_api_key)
#print('L3: Running Corporate Investment Module. Gathering Data...')
#corp_investment.to_csv('corporate_investment.csv')
#print('L3: Corporate Investment Module Complete.')