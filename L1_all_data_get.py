import pandas as pd
from API_keys import FRED_api_key, BEA_api_key, BLS_api_key

from L1_corp_profits_income_generators import get_corporate_profits
from L1_employment_lvl_income_generators import get_employment_level
from L1_govt_transfers_income_generators import get_govt_transfer_payments
from L1_wage_growth_income_generators import get_wage_growth

# L1 master file to get all income generator data

def L1_all_data_get(FRED_api_key, BEA_api_key, BLS_api_key):
    print('L1: Starting Layer 1 Data Collection...')

    corp_profits = get_corporate_profits(FRED_api_key, BEA_api_key)
    print('L1: Corporate Profits Complete.')

    employment = get_employment_level(FRED_api_key, BLS_api_key)
    print('L1: Employment Level Complete.')

    transfers = get_govt_transfer_payments(FRED_api_key)
    print('L1: Government Transfers Complete.')

    wages = get_wage_growth(FRED_api_key)
    print('L1: Wage Growth Complete.')

    L1_df = pd.concat([corp_profits, employment, transfers, wages], axis=1)
    L1_df.index.name = "date"

    print('L1: All Layer 1 Data Collection Complete.')

    return L1_df


#L1_data = L1_all_data_get(FRED_api_key, BEA_api_key)
#L1_data.to_csv('L1_data.csv')