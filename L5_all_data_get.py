import pandas as pd

from L5_spx500_data import get_equity_valuation

# L5 master file to get all S&P 500 equity valuation data

def L5_all_data_get():
    print('L5: Starting Layer 5 Data Collection...')

    equity_valuation = get_equity_valuation()
    print('L5: Equity Valuation Complete.')

    L5_df = pd.concat([equity_valuation], axis=1)
    L5_df.index.name = "date"

    print('L5: All Layer 5 Data Collection Complete.')

    return L5_df


#L5_data = L5_all_data_get()
#L5_data.to_csv('L5_data.csv')