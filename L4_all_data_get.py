import pandas as pd
from API_keys import FRED_api_key

from L4_corp_profits_prices_and_output import get_corporate_revenue
from L4_inflation_prices_and_output import get_inflation
from L4_production_prices_and_output import get_output_production

# L4 master file to get all prices and output data

def L4_all_data_get(FRED_api_key):
    print('L4: Starting Layer 4 Data Collection...')

    corporate_profits = get_corporate_revenue(FRED_api_key)
    print('L4: Corporate Profits Complete.')

    inflation = get_inflation(FRED_api_key)
    print('L4: Inflation Complete.')

    output_production = get_output_production(FRED_api_key)
    print('L4: Output & Production Complete.')

    L4_df = pd.concat([corporate_profits, inflation, output_production], axis=1)
    L4_df.index.name = "date"

    print('L4: All Layer 4 Data Collection Complete.')

    return L4_df


#L4_data = L4_all_data_get(FRED_api_key)
#L4_data.to_csv('L4_data.csv')