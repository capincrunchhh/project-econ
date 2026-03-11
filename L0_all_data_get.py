import pandas as pd
from API_keys import EIA_api_key, FRED_api_key

from L0_energy_constraints import get_all_energy_data
from L0_labor_constraints import get_labor_constraints
from L0_capital_availability_constraints import get_capital_constraints
from L0_tech_and_productivity_constraints import get_technology_constraints

# L0_master file to get all data

def L0_all_data_get(EIA_api_key, FRED_api_key):
    print('L0: Starting Layer 0 Data Collection...')

    energy = get_all_energy_data(EIA_api_key, FRED_api_key)
    print('L0: Energy Constraints Complete.')

    labor = get_labor_constraints(FRED_api_key)
    print('L0: Labor Constraints Complete.')

    capital = get_capital_constraints(FRED_api_key)
    print('L0: Capital Constraints Complete.')

    technology = get_technology_constraints(FRED_api_key)
    print('L0: Technology and Productivity Constraints Complete.')

    L0_df = pd.concat([energy, labor, capital, technology], axis=1)
    L0_df.index.name = "date"

    print('L0: All Layer 0 Data Collection Complete.')

    return L0_df

#L0_data = L0_all_data_get(EIA_api_key, FRED_api_key)
#L0_data.to_csv('L0_data.csv')

