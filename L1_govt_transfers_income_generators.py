import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key

# Transfer Payments and Disposable Income Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Transfer measures (FRED, quarterly → forward-fill to monthly):
# 1. W823RC1       social_benefits_yoy        social benefits to persons YoY        expressed as decimal. 5% = 0.05
# 2. A063RC1       current_transfers_yoy      current transfer payments YoY         expressed as decimal. 5% = 0.05
# 3. W825RC1       unemployment_benefits_yoy  unemployment benefits YoY             expressed as decimal. 5% = 0.05

# Disposable income (FRED, quarterly cadence → dropna + resample MS + ffill to monthly):
# 4. DPI           disposable_income_yoy      disposable personal income YoY        expressed as decimal. 5% = 0.05
# 5. DSPIC96       real_disposable_income_yoy real disposable personal income YoY   expressed as decimal. 5% = 0.05

# Derived fiscal intensity measure:
# 6. transfers_dpi_ratio                      social benefits / DPI level           expressed as decimal
#    note: NaN before 1959 (social_benefits starts 1959, dpi starts 1947)


def get_govt_transfer_payments(FRED_api_key):
    fred = Fred(FRED_api_key)
    # --- Quarterly series → forward-fill to monthly ---
    social_benefits       = fred.get_series("W823RC1").resample("MS").ffill()
    current_transfers     = fred.get_series("A063RC1").resample("MS").ffill()
    unemployment_benefits = fred.get_series("W825RC1").resample("MS").ffill()
    # --- Quarterly cadence published on FRED → dropna first then resample + ffill ---
    dpi      = fred.get_series("DPI").dropna().resample("MS").ffill()
    real_dpi = fred.get_series("DSPIC96").dropna().resample("MS").ffill()
    transfer_df = pd.DataFrame({
        "social_benefits":       social_benefits,
        "current_transfers":     current_transfers,
        "unemployment_benefits": unemployment_benefits,
        "dpi":                   dpi,
        "real_dpi":              real_dpi,
    })
    # --- YoY transformations ---
    transfer_df["social_benefits_yoy"]        = transfer_df["social_benefits"].pct_change(12, fill_method=None)
    transfer_df["current_transfers_yoy"]      = transfer_df["current_transfers"].pct_change(12, fill_method=None)
    transfer_df["unemployment_benefits_yoy"]  = transfer_df["unemployment_benefits"].pct_change(12, fill_method=None)
    transfer_df["disposable_income_yoy"]      = transfer_df["dpi"].pct_change(12, fill_method=None)
    transfer_df["real_disposable_income_yoy"] = transfer_df["real_dpi"].pct_change(12, fill_method=None)
    # --- Derived fiscal intensity ratio: social benefits / DPI ---
    # note: NaN before 1959 where social_benefits coverage begins
    transfer_df["transfers_dpi_ratio"] = transfer_df["social_benefits"] / transfer_df["dpi"]
    # --- Drop raw levels ---
    transfer_df = transfer_df.drop(columns=["social_benefits", "current_transfers",
                                             "unemployment_benefits", "dpi", "real_dpi"])
    transfer_df.index.name = "date"
    transfer_df = transfer_df.sort_index()
    transfer_df = transfer_df.loc[:transfer_df.last_valid_index()]

    return transfer_df





#
# RUNNING THE CODE:
#

#govt_transfer_payments = get_govt_transfer_payments(FRED_api_key)
#print('L1: Running Transfer Payments Module. Gathering Data...')
#govt_transfer_payments.to_csv('govt_transfer_payments.csv')
#print('L1: Transfer Payments Module Complete.')