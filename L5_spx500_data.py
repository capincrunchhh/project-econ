import pandas as pd
import numpy as np
import requests

# Equity Valuation Variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Source: Shiller/Yale data via posix4e JSON API (updated weekly, 1871-present)
# No API key required.

# 1. sp500              sp500_log          S&P 500 index log level             log(price level)   starts 1871
# 2. sp500              sp500_tr_yoy       S&P 500 total return YoY            expressed as decimal. 5% = 0.05  starts 1871
#    note: total return reconstructed from price + dividend reinvestment
# 3. earnings           sp500_eps_yoy      S&P 500 earnings per share YoY      expressed as decimal. 5% = 0.05  starts 1871
# 4. sp500 / earnings   sp500_pe           S&P 500 P/E ratio                   level  starts 1871, but is reliable after 1956
# 5. cape               cape               Shiller CAPE ratio                  level  starts 1881 (requires 10yr earnings history)
# 6. Derived            equity_risk_premium earnings yield minus 10yr rate     level (%)  starts 1871, but is reliable after 1956
#    note: equity_risk_premium = (earnings / sp500) - (long_interest_rate / 100)
#    earnings yield = 1 / PE, expressed as decimal
#    long_interest_rate from Shiller (GS10 equivalent) expressed as percent → divide by 100


SHILLER_URL = "https://posix4e.github.io/shiller_wrapper_data/data/stock_market_data.json"


def get_equity_valuation():
    # --- Pull Shiller JSON ---
    response = requests.get(SHILLER_URL)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date_string"])
    df = df.set_index("date").sort_index()
    # --- Select and rename relevant columns ---
    eq = df[["sp500", "earnings", "dividend", "cape", "long_interest_rate"]].copy()
    eq.columns = ["price", "earnings", "dividend", "cape", "gs10"]
    # --- Log level ---
    eq["sp500_log"] = np.log(eq["price"])
    # --- Total return index reconstruction (price + reinvested dividends) ---
    # dividend in Shiller is annualized → monthly dividend = dividend / 12
    # total return index: TR(t) = TR(t-1) * (price(t) + dividend(t)/12) / price(t-1)
    eq["monthly_dividend"] = eq["dividend"] / 12
    tr_index = [1.0]
    prices = eq["price"].values
    dividends = eq["monthly_dividend"].values
    for i in range(1, len(eq)):
        if pd.isna(prices[i]) or pd.isna(prices[i-1]) or prices[i-1] == 0:
            tr_index.append(np.nan)
        elif pd.isna(dividends[i]):
            # no dividend data — use price return only for recent months
            tr_index.append(tr_index[-1] * prices[i] / prices[i-1])
        else:
            tr_index.append(tr_index[-1] * (prices[i] + dividends[i]) / prices[i-1])
    eq["tr_index"] = tr_index
    eq["sp500_tr_yoy"] = eq["tr_index"].pct_change(12, fill_method=None)  # YoY total return
    eq["sp500_tr_6m"]  = eq["tr_index"].pct_change(6,  fill_method=None)  # 6-month total return
    eq["sp500_tr_3m"]  = eq["tr_index"].pct_change(3,  fill_method=None)  # 3-month total return
    eq["sp500_tr_1m"]  = eq["tr_index"].pct_change(1,  fill_method=None)  # month-over-month total return
    # --- EPS YoY ---
    eq["sp500_eps_yoy"] = eq["earnings"].pct_change(12, fill_method=None)
    # --- P/E ratio ---
    eq["sp500_pe"] = eq["price"] / eq["earnings"]
    # --- Equity risk premium: earnings yield minus 10yr rate, both in percent ---
    # earnings yield = (earnings / price) * 100 → converts to percent (e.g. 0.05 → 5.0)
    # gs10 already in percent (e.g. 4.0 = 4%)
    # result in percentage points: positive = equities cheap vs bonds, negative = bonds cheaper
    eq["equity_risk_premium"] = (eq["earnings"] / eq["price"] * 100) - eq["gs10"]
    # --- Drop intermediate columns ---
    eq = eq.drop(columns=["price", "earnings", "dividend", "gs10",
                           "monthly_dividend", "tr_index"])
    # --- Mask PE and ERP before 1956 (earnings series not comparable to price index pre-1956) ---
    eq.loc[eq.index < "1956-01-01", "sp500_pe"] = np.nan
    eq.loc[eq.index < "1956-01-01", "equity_risk_premium"] = np.nan
    # --- set final df
    eq.index.name = "date"
    eq = eq.sort_index()
    eq = eq.loc[:eq.last_valid_index()]

    return eq





#
# RUNNING THE CODE:
#

#equity_valuation = get_equity_valuation()
#print('L5: Running Equity Valuation Module. Gathering Data...')
#equity_valuation.to_csv('equity_valuation.csv')
#print('L5: Equity Valuation Module Complete.')