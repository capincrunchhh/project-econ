import pandas as pd
from API_keys import FRED_api_key

from L2_corp_leverage_balance_sheet import get_corporate_leverage
from L2_financial_conditions_balance_sheet import get_financial_conditions
from L2_household_assets_balance_sheet import get_household_balance_sheet
from L2_household_debt_balance_sheet import get_household_debt

# L2 master file to get all balance sheet data

def L2_all_data_get(FRED_api_key):
    print('L2: Starting Layer 2 Data Collection...')

    corp_leverage = get_corporate_leverage(FRED_api_key)
    print('L2: Corporate Leverage Complete.')

    financial_conditions = get_financial_conditions(FRED_api_key)
    print('L2: Financial Conditions Complete.')

    household_assets = get_household_balance_sheet(FRED_api_key)
    print('L2: Household Assets Complete.')

    household_debt = get_household_debt(FRED_api_key)
    print('L2: Household Debt Complete.')

    L2_df = pd.concat([corp_leverage, financial_conditions, household_assets, household_debt], axis=1)
    L2_df.index.name = "date"

    print('L2: All Layer 2 Data Collection Complete.')

    return L2_df


#L2_data = L2_all_data_get(FRED_api_key)
#L2_data.to_csv('L2_data.csv')