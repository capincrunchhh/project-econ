import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key

# Wage and Income Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Nominal wage growth:
# 1. AHETPI         ahe_yoy            avg hourly earnings YoY             expressed as decimal. 5% = 0.05  monthly
# 2. ECIWAG         eci_yoy            employment cost index YoY           expressed as decimal. 5% = 0.05  quarterly forward-filled
# 3. COMPNFB        comp_per_hour_yoy  compensation per hour YoY           expressed as decimal. 5% = 0.05  quarterly forward-filled
# 4. LES1252881600Q median_weekly_yoy  median weekly earnings YoY          expressed as decimal. 5% = 0.05  quarterly forward-filled


def get_wage_growth(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Monthly series ---
    ahe = fred.get_series("AHETPI")
    # --- Quarterly series → forward-fill to monthly ---
    eci           = fred.get_series("ECIWAG").resample("MS").ffill()
    comp          = fred.get_series("COMPNFB").resample("MS").ffill()
    median_weekly = fred.get_series("LES1252881600Q").resample("MS").ffill()
    wage_df = pd.DataFrame({
        "ahe":           ahe,
        "eci":           eci,
        "comp_per_hour": comp,
        "median_weekly": median_weekly,
    })
    # --- Nominal YoY ---
    wage_df["ahe_yoy"]           = wage_df["ahe"].pct_change(12, fill_method=None)
    wage_df["eci_yoy"]           = wage_df["eci"].pct_change(12, fill_method=None)
    wage_df["comp_per_hour_yoy"] = wage_df["comp_per_hour"].pct_change(12, fill_method=None)
    wage_df["median_weekly_yoy"] = wage_df["median_weekly"].pct_change(12, fill_method=None)
    # --- Drop raw levels ---
    wage_df = wage_df.drop(columns=["ahe", "eci", "comp_per_hour", "median_weekly"])
    wage_df.index.name = "date"
    wage_df = wage_df.sort_index()
    wage_df = wage_df.loc[:wage_df.last_valid_index()]

    return wage_df





#
# RUNNING THE CODE:
#

#wage_growth = get_wage_growth(FRED_api_key)
#print('L1: Running Wage Growth Module. Gathering Data...')
#wage_growth.to_csv('wage_growth.csv')
#print('L1: Wage Growth Module Complete.')