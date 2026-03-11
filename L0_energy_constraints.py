import requests
import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key, EIA_api_key

# 9 Core energy variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Energy mix %:
# 1a. petroleum_share          petroleum % of total energy mix             expressed as decimal. 5% = 0.05
# 1b. natural_gas_share        natural gas % of total energy mix           expressed as decimal. 5% = 0.05
# 1c. coal_share               coal % of total energy mix                  expressed as decimal. 5% = 0.05
# 1d. renewables_share         renewables % of total energy mix            expressed as decimal. 5% = 0.05
# 1e. nuclear_share            nuclear % of total energy mix               expressed as decimal. 5% = 0.05

# Current burden:
# 2. MCOILWTICO               WTI crude oil price                          expressed as USD per barrel
# 3. GASREGM                  retail gasoline price                        expressed as USD per gallon
# 4. CPIENGSL                 energy CPI                                   expressed as index value (1982-84=100)

# Forward pressure:
# 5. MCRFPUS2                US crude production                           expressed as thousands of barrels per day
# 6. WTTSTUS1                US crude inventories (monthly avg)            expressed as thousands of barrels
# 7. inventory_change_rate   month-over-month inventory change             expressed as decimal. 5% = 0.05

# Shock risk:
# 8. oil_volatility          6-month rolling std of oil price pct change   expressed as decimal
# 9. inventory_consumption_ratio  crude inventories / consumption          expressed as days of supply


# Generic function to get EIA series
def get_eia_series(url, params, EIA_api_key):
    params["api_key"] = EIA_api_key
    response = requests.get(url, params=params)
    response.raise_for_status()
    json_data = response.json()
    if "response" not in json_data:
        raise ValueError(f"Unexpected API response: {json_data}")
    data = json_data["response"]["data"]
    eia_df = pd.DataFrame(data)
    if eia_df.empty:
        raise ValueError("No data returned. Check URL parameters.")
    eia_df["period"] = pd.to_datetime(eia_df["period"])
    if "value" in eia_df.columns:
        eia_df["value"] = pd.to_numeric(eia_df["value"], errors="coerce")
    # Pivot if multiple series present
    if "msn" in eia_df.columns:
        eia_df = eia_df.pivot(index="period", columns="msn", values="value")
    else:
        eia_df = eia_df.set_index("period")

    return eia_df.sort_index()


# function to get US energy mix from EIA, this pulls in generic function
def get_us_energy_mix(EIA_api_key):
    # ALL UNITS ARE IN TRILLIONS OF BPUS
    msn_codes = ["TETCBUS", "PMTCBUS", "NNTCBUS", "CLTCBUS", "RETCBUS", "NUETBUS"]
    url = "https://api.eia.gov/v2/total-energy/data/"
    params = {
        "frequency": "monthly",
        "data[0]": "value",
        "facets[msn][]": msn_codes,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    df_raw = get_eia_series(url, params, EIA_api_key)
    # Rename columns to human-readable names
    df_raw = df_raw.rename(columns={
        "TETCBUS": "total_energy",
        "PMTCBUS": "petroleum",
        "NNTCBUS": "natural_gas",
        "CLTCBUS": "coal",
        "RETCBUS": "renewables",
        "NUETBUS": "nuclear"
    })
    # Compute percentage mix
    energy_mix_pct = df_raw.div(df_raw["total_energy"], axis=0)
    # rename columns
    energy_mix_pct = energy_mix_pct.rename(columns={
        "petroleum": "petroleum_share",
        "natural_gas": "natural_gas_share",
        "coal": "coal_share",
        "renewables": "renewables_share",
        "nuclear": "nuclear_share"
    })
    # trims NaN rows from bottom
    #energy_mix_pct = energy_mix_pct.loc[:energy_mix_pct.last_valid_index()]

    return energy_mix_pct


# function to get Current Burden FRED data
def get_energy_burden(FRED_api_key):
    fred = Fred(FRED_api_key)
    fred_df = pd.DataFrame({
        "oil_price": fred.get_series("MCOILWTICO"),
        "gas_price": fred.get_series("GASREGM"),
        "energy_cpi": fred.get_series("CPIENGSL")
    })

    return fred_df


# function to get Forward Pressure EIA data
def get_forward_pressure(EIA_api_key):
    # --- Crude Production (monthly, snd endpoint) ---
    url1 = "https://api.eia.gov/v2/petroleum/sum/snd/data/"
    prod_params = {
        "frequency": "monthly",
        "data[0]": "value",
        "facets[series][]": "MCRFPUS2", # UNITS IN THOUSANDS OF BARRELS PRODUCED PER DAY (MONTHLY RATE)
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    prod_df = get_eia_series(url1, prod_params, EIA_api_key)
    prod_df = prod_df[["value"]].rename(columns={"value": "crude_production"})
    prod_df["crude_production"] = pd.to_numeric(prod_df["crude_production"], errors="coerce")
    # --- Crude Inventories (weekly, sndw endpoint) → resample to monthly ---
    url2 = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
    inv_params = {
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WTTSTUS1", # UNITS IN THOUSANDS OF BARRELS
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    inv_df = get_eia_series(url2, inv_params, EIA_api_key)
    inv_df = inv_df[["value"]].rename(columns={"value": "crude_inventories"})
    inv_df["crude_inventories"] = pd.to_numeric(inv_df["crude_inventories"], errors="coerce")
    inv_df = inv_df.resample("MS").mean()
    # --- Combine and construct inventory % change rate ---
    fwd_pressure_df = pd.concat([prod_df, inv_df], axis=1)
    fwd_pressure_df["inventory_change_rate"] = fwd_pressure_df["crude_inventories"].pct_change()

    return fwd_pressure_df


def get_shock_risk(FRED_api_key, EIA_api_key):
    # --- Oil Price Volatility (FRED) ---
    fred = Fred(FRED_api_key)
    shock_df = pd.DataFrame({
        "oil_volatility": fred.get_series("MCOILWTICO").pct_change().rolling(6).std()
    })
    # --- Crude Inventories (weekly → monthly) ---
    url1 = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
    inv_params = {
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WTTSTUS1",  # UNITS IN THOUSANDS OF BARRELS
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    inv_df = get_eia_series(url1, inv_params, EIA_api_key)
    inv_df = inv_df[["value"]].rename(columns={"value": "crude_inventories"})
    inv_df["crude_inventories"] = pd.to_numeric(inv_df["crude_inventories"], errors="coerce")
    inv_df = inv_df.resample("MS").mean()
    # --- Crude Consumption (monthly) ---
    url2 = "https://api.eia.gov/v2/petroleum/sum/snd/data/"
    con_params = {
        "frequency": "monthly",
        "data[0]": "value",
        "facets[series][]": "MTTUPUS2",  # UNITS IN THOUSANDS OF BARRELS PER DAY (MONTHLY RATE)
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    con_df = get_eia_series(url2, con_params, EIA_api_key)
    con_df = con_df[["value"]].rename(columns={"value": "crude_consumption"})
    con_df["crude_consumption"] = pd.to_numeric(con_df["crude_consumption"], errors="coerce")
    # --- Inventory-to-Consumption Ratio ---
    ratio_df = pd.concat([inv_df, con_df], axis=1)
    ratio_df["inventory_consumption_ratio"] = ratio_df["crude_inventories"] / ratio_df["crude_consumption"]
    # --- Merge all ---
    shock_df = pd.concat([shock_df, ratio_df], axis=1)
    shock_df = shock_df.loc[:shock_df.last_valid_index()]
    # --- Drop unused ---
    shock_df = shock_df.drop(columns=["crude_inventories", "crude_consumption"])

    return shock_df


def get_all_energy_data(EIA_api_key, FRED_api_key):
    energy_mix = get_us_energy_mix(EIA_api_key)
    burden = get_energy_burden(FRED_api_key)
    fwd_pressure = get_forward_pressure(EIA_api_key)
    shock_risk = get_shock_risk(FRED_api_key, EIA_api_key)

    energy_constraints_df = pd.concat([energy_mix, burden, fwd_pressure, shock_risk], axis=1)
    energy_constraints_df.index.name = "date"
    energy_constraints_df = energy_constraints_df.drop(columns=['total_energy'])

    return energy_constraints_df




#
# RUNNING THE CODE:
#

#energy_constraints = get_all_energy_data(EIA_api_key, FRED_api_key)
#print('L0: Running Energy Constraints Module. Gathering Data...')
#energy_constraints.to_csv('energy_module_data.csv')
#print('L0: Energy Constraints Module Complete.')





# MISC Links:
#CRUDE PRODUCTION: https://www.eia.gov/opendata/browser/petroleum/sum/snd?frequency=monthly&data=value;&facets=series;&series=MCRFPUS2;&sortColumn=period;&sortDirection=desc;
#CRUDE SUPPLY: https://www.eia.gov/opendata/browser/petroleum/sum/sndw?frequency=weekly&data=value;&facets=series;&series=WTTSTUS1;&sortColumn=period;&sortDirection=desc;
#CRUDE CONSUMPTION: https://www.eia.gov/opendata/browser/petroleum/sum/snd?frequency=monthly&data=value;&facets=series;&series=MTTUPUS2;&sortColumn=period;&sortDirection=desc;
