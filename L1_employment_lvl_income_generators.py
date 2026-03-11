import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from API_keys import BLS_api_key, FRED_api_key

# Employment Level Variables. ALL DATA MONTHLY.

# Core employment (FRED):
# 1. PAYEMS          total_payroll_yoy              total nonfarm payrolls YoY          expressed as decimal. 5% = 0.05
# 2. PAYEMS          total_payroll_log              total nonfarm payrolls log level    expressed as log(level)
# 3. W875RX1         real_aggregate_payrolls_yoy    real aggregate payrolls YoY         expressed as decimal. 5% = 0.05
# 4. HOANBS          total_hours_worked_yoy         total hours worked YoY              expressed as decimal. 5% = 0.05

# Core employment (BLS):
# 5. CES0500000001   private_payrolls_yoy           private payrolls YoY                expressed as decimal. 5% = 0.05  starts 1939
# 6. LNS12000000     household_employment_yoy       household employment YoY            expressed as decimal. 5% = 0.05  starts 1948
# 7. CES0500000006   nominal_aggregate_payrolls_yoy nominal aggregate payrolls YoY      expressed as decimal. 5% = 0.05  starts 1964

# Sector employment YoY (BLS) — all expressed as decimal. 5% = 0.05, all start 1939
# 8.  CES0000000001  total_nonfarm_yoy                  expressed as decimal. 5% = 0.05, all start 1939
# 9.  CES0500000001  private_yoy                        expressed as decimal. 5% = 0.05, all start 1939
# 10. CES1000000001  mining_logging_yoy                 expressed as decimal. 5% = 0.05, all start 1939
# 11. CES2000000001  construction_yoy                   expressed as decimal. 5% = 0.05, all start 1939
# 12. CES3000000001  manufacturing_yoy                  expressed as decimal. 5% = 0.05, all start 1939
# 13. CES4000000001  trade_transport_utilities_yoy      expressed as decimal. 5% = 0.05, all start 1939
# 14. CES5000000001  information_yoy                    expressed as decimal. 5% = 0.05, all start 1939
# 15. CES5500000001  financial_activities_yoy           expressed as decimal. 5% = 0.05, all start 1939
# 16. CES6000000001  professional_business_yoy          expressed as decimal. 5% = 0.05, all start 1939
# 17. CES6500000001  education_health_yoy               expressed as decimal. 5% = 0.05, all start 1939
# 18. CES7000000001  leisure_hospitality_yoy            expressed as decimal. 5% = 0.05, all start 1939
# 19. CES8000000001  other_services_yoy                 expressed as decimal. 5% = 0.05, all start 1939
# 20. CES9000000001  government_yoy                     expressed as decimal. 5% = 0.05, all start 1939


# Generic BLS API fetch function — chunks requests into 20-year windows
def get_bls_series(series_ids, BLS_api_key, start_year):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    end_year = pd.Timestamp.now().year
    all_dfs = []
    for chunk_start in range(start_year, end_year + 1, 20):
        chunk_end = min(chunk_start + 19, end_year)
        payload = {
            "seriesid": series_ids,
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
            "registrationkey": BLS_api_key
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        if json_data["status"] != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS API error: {json_data['message']}")
        dfs = []
        for series in json_data["Results"]["series"]:
            series_id = series["seriesID"]
            rows = []
            for item in series["data"]:
                if item["period"].startswith("M") and item["period"] != "M13":
                    if item["value"] == "-":
                        continue
                    date = pd.to_datetime(f"{item['year']}-{item['period'][1:]}-01")
                    rows.append({"date": date, series_id: float(item["value"].replace(",", ""))})
            if rows:
                df = pd.DataFrame(rows).set_index("date")
                dfs.append(df)
        if dfs:
            all_dfs.append(pd.concat(dfs, axis=1))
    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    return combined


def get_employment_level(FRED_api_key, BLS_api_key):
    fred = Fred(FRED_api_key)

    # --- FRED core series ---
    # Resample HOANBS to monthly before building dataframe to avoid quarterly gaps
    total_hours = fred.get_series("HOANBS").resample("MS").ffill()

    fred_df = pd.DataFrame({
        "total_payroll":           fred.get_series("PAYEMS"),
        "real_aggregate_payrolls": fred.get_series("W875RX1"),
        "total_hours_worked":      total_hours,
    })
    fred_df["total_payroll_yoy"]           = fred_df["total_payroll"].pct_change(12, fill_method=None)
    fred_df["total_payroll_log"]           = np.log(fred_df["total_payroll"])
    fred_df["real_aggregate_payrolls_yoy"] = fred_df["real_aggregate_payrolls"].pct_change(12, fill_method=None)
    fred_df["total_hours_worked_yoy"]      = fred_df["total_hours_worked"].pct_change(12, fill_method=None)
    fred_df = fred_df.drop(columns=["total_payroll", "real_aggregate_payrolls", "total_hours_worked"])

    # --- BLS core series --- pulled separately due to different start years
    bls_early    = get_bls_series(["CES0500000001", "LNS12000000"], BLS_api_key, start_year=1939)
    bls_payrolls = get_bls_series(["CES0500000006"], BLS_api_key, start_year=1964)
    bls_core = pd.concat([bls_early, bls_payrolls], axis=1)
    bls_core = bls_core.rename(columns={
        "CES0500000001": "private_payrolls",
        "LNS12000000":   "household_employment",
        "CES0500000006": "nominal_aggregate_payrolls",
    })
    bls_core["private_payrolls_yoy"]           = bls_core["private_payrolls"].pct_change(12, fill_method=None)
    bls_core["household_employment_yoy"]       = bls_core["household_employment"].pct_change(12, fill_method=None)
    bls_core["nominal_aggregate_payrolls_yoy"] = bls_core["nominal_aggregate_payrolls"].pct_change(12, fill_method=None)
    bls_core = bls_core.drop(columns=["private_payrolls", "household_employment", "nominal_aggregate_payrolls"])

    # --- BLS sector series --- all start 1939
    sector_ids = [
        "CES0000000001", "CES0500000001", "CES1000000001", "CES2000000001",
        "CES3000000001", "CES4000000001", "CES5000000001", "CES5500000001",
        "CES6000000001", "CES6500000001", "CES7000000001", "CES8000000001",
        "CES9000000001"
    ]
    bls_sectors = get_bls_series(sector_ids, BLS_api_key, start_year=1939)
    bls_sectors = bls_sectors.rename(columns={
        "CES0000000001": "total_nonfarm",
        "CES0500000001": "private",
        "CES1000000001": "mining_logging",
        "CES2000000001": "construction",
        "CES3000000001": "manufacturing",
        "CES4000000001": "trade_transport_utilities",
        "CES5000000001": "information",
        "CES5500000001": "financial_activities",
        "CES6000000001": "professional_business",
        "CES6500000001": "education_health",
        "CES7000000001": "leisure_hospitality",
        "CES8000000001": "other_services",
        "CES9000000001": "government",
    })
    for col in bls_sectors.columns:
        bls_sectors[f"{col}_yoy"] = bls_sectors[col].pct_change(12, fill_method=None)
    bls_sectors = bls_sectors.drop(columns=[c for c in bls_sectors.columns if not c.endswith("_yoy")])

    # --- Combine all ---
    employment_df = pd.concat([fred_df, bls_core, bls_sectors], axis=1)
    employment_df.index.name = "date"
    employment_df = employment_df.sort_index()
    employment_df = employment_df.loc[:employment_df.last_valid_index()]

    return employment_df





#
# RUNNING THE CODE:
#

#employment_level = get_employment_level(FRED_api_key, BLS_api_key)
#print('L1: Running Employment Level Module. Gathering Data...')
#employment_level.to_csv('employment_level.csv')
#print('L1: Employment Level Module Complete.')