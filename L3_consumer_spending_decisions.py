import pandas as pd
import numpy as np
from fredapi import Fred
from API_keys import FRED_api_key

# Consumer Spending Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Core consumption (FRED, monthly):
# 1. PCE/PCEPI               pce_yoy                    real PCE YoY                              expressed as decimal. 5% = 0.05
# 2. PCESV/DSERRG3M086SBEA   pce_services_yoy           real PCE services YoY                     expressed as decimal. 5% = 0.05
# 3. PCEDG/DGDSRG3M086SBEA   pce_goods_yoy              real PCE goods YoY                        expressed as decimal. 5% = 0.05
# 4. CMRMTSPL                retail_sales_yoy           real retail sales YoY                     expressed as decimal. 5% = 0.05
# 5. RRSFS                   core_retail_sales_yoy      real core retail sales (control) YoY      expressed as decimal. 5% = 0.05

# Spending capacity:
# 6. DSPIC96     real_dpi_yoy               real disposable personal income YoY                   expressed as decimal. 5% = 0.05  quarterly forward-filled
# 7. TOTALSL     consumer_credit_yoy        total consumer credit YoY                             expressed as decimal. 5% = 0.05

# Saving behavior:
# 8. PSAVERT     personal_saving_rate       personal saving rate                                  expressed as level (6.0 = 6.0%)
# 9. PSAVERT     personal_saving_rate_yoy   personal saving rate YoY change                       expressed as decimal. 5% = 0.05

# Derived sustainability ratios (level):
# 10. derived    spending_sustainability_ratio   PCE / DPI                                        expressed as ratio level
#     note: deflated PCE in billions chained, DSPIC96 in billions chained → no unit adjustment
# 11. derived    consumer_credit_pce_ratio       consumer credit / PCE                            expressed as ratio level
#     note: TOTALSL in millions, deflated PCE in billions chained → divide TOTALSL by 1000
# 12. derived    net_worth_dpi_ratio             household net worth / DPI                        expressed as ratio level
#     note: BOGZ1FL192090005Q in millions, DSPIC96 in billions chained → divide net worth by 1000
#     quarterly forward-filled
# 13. RETAILIRSA retail_inventory_sales_ratio    retail inventories to sales ratio                expressed as ratio level (monthly)
# 14. derived    durables_pce_share              PCE durables / total PCE                         expressed as ratio level
#     note: PCEDG quarterly forward-filled, PCE monthly → both nominal for consistent ratio


def get_consumer_spending(FRED_api_key):
    fred = Fred(FRED_api_key)

    # --- Monthly series ---
    retail_sales         = fred.get_series("CMRMTSPL")
    core_retail          = fred.get_series("RRSFS")
    consumer_credit      = fred.get_series("TOTALSL")
    saving_rate          = fred.get_series("PSAVERT")
    retail_inv_sales     = fred.get_series("RETAILIRSA")

    # --- Real PCE: deflate nominal by chain-type price indexes → back to 1959 ---
    pce_nominal       = fred.get_series("PCE").dropna().resample("MS").ffill()
    pce_deflator      = fred.get_series("PCEPI")
    pce               = (pce_nominal / pce_deflator) * 100

    services_deflator = fred.get_series("DSERRG3M086SBEA")
    goods_deflator    = fred.get_series("DGDSRG3M086SBEA")
    pce_services      = (fred.get_series("PCESV").dropna().resample("MS").ffill() / services_deflator) * 100
    pce_goods         = (fred.get_series("PCEDG").dropna().resample("MS").ffill() / goods_deflator) * 100

    # --- Nominal PCE durables for share ratio (keep nominal for consistent ratio with nominal PCE) ---
    pce_durables_nom  = fred.get_series("PCEDG").dropna().resample("MS").ffill()

    # --- Quarterly → dropna + forward-fill to monthly ---
    real_dpi          = fred.get_series("DSPIC96").dropna().resample("MS").ffill()
    net_worth         = fred.get_series("BOGZ1FL192090005Q").dropna().resample("MS").ffill()

    df = pd.DataFrame({
        "pce":                  pce_nominal,        # nominal for durables share ratio
        "pce_real":             pce,                # real for YoY and sustainability ratio
        "pce_services":         pce_services,
        "pce_goods":            pce_goods,
        "pce_durables_nom":     pce_durables_nom,
        "retail_sales":         retail_sales,
        "core_retail":          core_retail,
        "real_dpi":             real_dpi,
        "consumer_credit":      consumer_credit,
        "saving_rate":          saving_rate,
        "net_worth":            net_worth,
        "retail_inv_sales":     retail_inv_sales,
    })

    # --- YoY transformations ---
    df["pce_yoy"]                  = df["pce_real"].pct_change(12, fill_method=None)
    df["pce_services_yoy"]         = df["pce_services"].pct_change(12, fill_method=None)
    df["pce_goods_yoy"]            = df["pce_goods"].pct_change(12, fill_method=None)
    df["retail_sales_yoy"]         = df["retail_sales"].pct_change(12, fill_method=None)
    df["core_retail_sales_yoy"]    = df["core_retail"].pct_change(12, fill_method=None)
    df["real_dpi_yoy"]             = df["real_dpi"].pct_change(12, fill_method=None)
    df["consumer_credit_yoy"]      = df["consumer_credit"].pct_change(12, fill_method=None)
    df["personal_saving_rate_yoy"] = df["saving_rate"].pct_change(12, fill_method=None)

    # --- Levels ---
    df["personal_saving_rate"]         = df["saving_rate"]
    df["retail_inventory_sales_ratio"] = df["retail_inv_sales"]

    # --- Derived ratios (level) ---
    # deflated PCE and DSPIC96 both in billions chained → no unit adjustment
    df["spending_sustainability_ratio"] = df["pce_real"]           / df["real_dpi"]
    # TOTALSL in millions, deflated PCE in billions chained → divide TOTALSL by 1000
    df["consumer_credit_pce_ratio"]     = (df["consumer_credit"] / 1000) / df["pce_real"]
    # net worth in millions, DSPIC96 in billions chained → divide net worth by 1000
    df["net_worth_dpi_ratio"]           = (df["net_worth"] / 1000) / df["real_dpi"]
    # both nominal → consistent ratio
    df["durables_pce_share"]            = df["pce_durables_nom"]   / df["pce"]

    # --- Drop raw levels ---
    df = df.drop(columns=["pce", "pce_real", "pce_services", "pce_goods", "pce_durables_nom",
                           "retail_sales", "core_retail", "real_dpi", "consumer_credit",
                           "saving_rate", "net_worth", "retail_inv_sales"])

    df.index.name = "date"
    df = df.sort_index()
    df = df.loc[:df.last_valid_index()]

    return df





#
# RUNNING THE CODE:
#

#consumer_spending = get_consumer_spending(FRED_api_key)
#print('L3: Running Consumer Spending Module. Gathering Data...')
#consumer_spending.to_csv('consumer_spending.csv')
#print('L3: Consumer Spending Module Complete.')