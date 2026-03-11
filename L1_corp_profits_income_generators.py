import pandas as pd
import numpy as np
from fredapi import Fred
import beaapi
from API_keys import FRED_api_key, BEA_api_key

# Corporate Profit and Margin Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Core profit measures (FRED, quarterly → forward-fill to monthly):
# 1. CP            corp_profits_aftertax_yoy  corporate profits after tax YoY       expressed as decimal. 5% = 0.05
# 2. CPATAX        corp_profits_pretax_yoy    corporate profits before tax YoY      expressed as decimal. 5% = 0.05
# 3. NFCPATAX      nonfinancial_profits_yoy   nonfinancial corporate profits YoY    expressed as decimal. 5% = 0.05
# 4. CNCF          corp_cashflow_yoy          corporate cash flow YoY               expressed as decimal. 5% = 0.05
# 5. GDI           gdi_yoy                    gross domestic income YoY             expressed as decimal. 5% = 0.05

# Margin measures (derived, quarterly → forward-fill to monthly):
# 6. profit_margin                            CP / GDP level                        expressed as decimal
# 7. profit_margin_yoy                        profit margin YoY change              expressed as decimal. 5% = 0.05

# BEA NIPA diagnostic:
# 8. B179RC        iva_yoy                    inventory valuation adjustment YoY    expressed as decimal. 5% = 0.05  quarterly forward-filled

# Leading indicators (FRED, monthly):
# 9.  AMTMNO       mfg_new_orders_yoy         manufacturers new orders YoY          expressed as decimal. 5% = 0.05
# 10. CMRMTSPL     retail_sales_yoy           real retail and food services YoY     expressed as decimal. 5% = 0.05


def get_corporate_profits(FRED_api_key, BEA_api_key):
    fred = Fred(FRED_api_key)
    # --- Quarterly series → forward-fill to monthly ---
    cp       = fred.get_series("CP").resample("MS").ffill()
    cpatax   = fred.get_series("CPATAX").resample("MS").ffill()
    nfcpatax = fred.get_series("NFCPATAX").resample("MS").ffill()
    cncf     = fred.get_series("CNCF").resample("MS").ffill()
    gdi      = fred.get_series("GDI").resample("MS").ffill()
    gdp      = fred.get_series("GDP").resample("MS").ffill()
    # --- Monthly series ---
    mfg_new_orders = fred.get_series("AMTMNO")
    retail_sales   = fred.get_series("CMRMTSPL")
    # --- BEA NIPA: Inventory Valuation Adjustment ---
    bea_data = beaapi.get_data(
        BEA_api_key,
        datasetname="NIPA",
        TableName="T11200",
        Frequency="Q",
        Year="ALL"
    )
    iva_raw = bea_data[
        (bea_data["LineDescription"] == "Inventory valuation adjustment") &
        (bea_data["SeriesCode"] == "B179RC")
    ].copy()
    iva_raw["date"]  = pd.PeriodIndex(iva_raw["TimePeriod"], freq="Q").to_timestamp()
    iva_raw["value"] = pd.to_numeric(iva_raw["DataValue"], errors="coerce")
    iva = iva_raw.set_index("date")[["value"]].rename(columns={"value": "iva"})
    iva = iva.sort_index().resample("MS").ffill()
    profit_df = pd.DataFrame({
        "cp":              cp,
        "cpatax":          cpatax,
        "nfcpatax":        nfcpatax,
        "cncf":            cncf,
        "gdi":             gdi,
        "gdp":             gdp,
        "iva":             iva["iva"],
        "mfg_new_orders":  mfg_new_orders,
        "retail_sales":    retail_sales,
    })
    # --- YoY transformations ---
    profit_df["corp_profits_aftertax_yoy"] = profit_df["cp"].pct_change(12, fill_method=None)
    profit_df["corp_profits_pretax_yoy"]   = profit_df["cpatax"].pct_change(12, fill_method=None)
    profit_df["nonfinancial_profits_yoy"]  = profit_df["nfcpatax"].pct_change(12, fill_method=None)
    profit_df["corp_cashflow_yoy"]         = profit_df["cncf"].pct_change(12, fill_method=None)
    profit_df["gdi_yoy"]                   = profit_df["gdi"].pct_change(12, fill_method=None)
    profit_df["iva_yoy"]                   = profit_df["iva"].pct_change(12, fill_method=None)
    profit_df["mfg_new_orders_yoy"]        = profit_df["mfg_new_orders"].pct_change(12, fill_method=None)
    profit_df["retail_sales_yoy"]          = profit_df["retail_sales"].pct_change(12, fill_method=None)
    # --- Profit margin (CP / GDP) ---
    profit_df["profit_margin"]     = profit_df["cp"] / profit_df["gdp"]
    profit_df["profit_margin_yoy"] = profit_df["profit_margin"].pct_change(12, fill_method=None)
    # --- Drop raw levels ---
    profit_df = profit_df.drop(columns=["cp", "cpatax", "nfcpatax", "cncf", "gdi", "gdp", "iva",
                                         "mfg_new_orders", "retail_sales"])
    profit_df.index.name = "date"
    profit_df = profit_df.sort_index()
    profit_df = profit_df.loc[:profit_df.last_valid_index()]

    return profit_df




#
# RUNNING THE CODE:
#

#corporate_profits = get_corporate_profits(FRED_api_key, BEA_api_key)
#print('L1: Running Corporate Profits Module. Gathering Data...')
#corporate_profits.to_csv('corporate_profits.csv')
#print('L1: Corporate Profits Module Complete.')