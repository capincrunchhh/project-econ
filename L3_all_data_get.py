import pandas as pd
from API_keys import FRED_api_key

from L3_business_investment_spending_decisions import get_corporate_investment
from L3_consumer_spending_decisions import get_consumer_spending
from L3_govt_spending_decisions import get_government_spending

# L3 master file to get all spending decisions data

def L3_all_data_get(FRED_api_key):
    print('L3: Starting Layer 3 Data Collection...')

    corporate_investment = get_corporate_investment(FRED_api_key)
    print('L3: Corporate Investment Complete.')

    consumer_spending = get_consumer_spending(FRED_api_key)
    print('L3: Consumer Spending Complete.')

    govt_spending = get_government_spending(FRED_api_key)
    print('L3: Government Spending Complete.')

    L3_df = pd.concat([corporate_investment, consumer_spending, govt_spending], axis=1)
    L3_df.index.name = "date"

    print('L3: All Layer 3 Data Collection Complete.')

    return L3_df


#L3_data = L3_all_data_get(FRED_api_key, BEA_api_key)
#L3_data.to_csv('L3_data.csv')