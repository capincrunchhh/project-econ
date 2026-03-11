import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key

# 11 capital availability variables (all FRED data):

# Cost of capital:
# 1. DGS10                    10Y: nominal long-duration capital cost        expressed as 1.05 = 1.05%
# 2. DFII10                   Tips_10Y: real long-duration capital cost      expressed as 1.05 = 1.05%
# 3. real_fed_funds           real short-term capital cost                   expressed as 1.05 = 1.05%

# Credit risk and availability:
# 4. BAMLC0A0CM               investment-grade credit spread                 expressed as 1.05 = 1.05%
# 5. BAMLH0A0HYM2             high-yield credit spread                       expressed as 1.05 = 1.05%

# Bank lending capacity:
# 6. DPSACBW027SBOG YoY       bank deposit growth                            expressed as decimal. 5% = .05
# 7. TOTCI YoY                commercial credit growth                       expressed as decimal. 5% = .05
# 8. DRTSCILM                 lending standards                              expressed as index value (net % of banks tightening)

# System liquidity supply:
# 9. TOTRESNS YoY             bank reserve growth                            expressed as decimal. 5% = .05
# 10. WALCL                   Fed balance sheet size                         expressed as Millions USD
# 11. NFCI                    financial conditions index                     expressed as index value (negative = accommodative)


def get_capital_constraints(FRED_api_key):
    fred = Fred(FRED_api_key)
    # Real fed funds = nominal fed funds - CPI inflation
    fed_funds = fred.get_series("FEDFUNDS")                    # monthly
    cpi = fred.get_series("CPIAUCSL").pct_change(12, fill_method=None) * 100     # monthly
    real_fed_funds = fed_funds - cpi
    capital_df = pd.DataFrame({
        "treasury_10y":         fred.get_series("DGS10").resample("MS").mean(),           # Resampled to Monthly (daily)
        "tips_10y":             fred.get_series("DFII10").resample("MS").mean(),           # Resampled to Monthly (daily)
        "real_fed_funds":       real_fed_funds,                                            # monthly
        "ig_credit_spread":     fred.get_series("BAMLC0A0CM").resample("MS").mean(),      # Resampled to Monthly (daily)
        "hy_credit_spread":     fred.get_series("BAMLH0A0HYM2").resample("MS").mean(),    # Resampled to Monthly (daily)
        "bank_deposits":        fred.get_series("DPSACBW027SBOG").resample("MS").mean(),  # Resampled to Monthly (weekly)
        "commercial_credit": fred.get_series("TOTCI").resample("MS").mean(),  # Resampled to Monthly (weekly)
        "lending_standards":    fred.get_series("DRTSCILM").resample("MS").ffill(),       # Resampled to Monthly (quarterly)
        "bank_reserves":        fred.get_series("TOTRESNS").resample("MS").mean(),        # Resampled to Monthly (weekly)
        "fed_balance_sheet":    fred.get_series("WALCL").resample("MS").mean(),           # Resampled to Monthly (weekly)
        "financial_conditions": fred.get_series("NFCI").resample("MS").mean(),            # Resampled to Monthly (weekly)
    })
    # YoY transformations
    capital_df["bank_deposits_yoy"] = capital_df["bank_deposits"].pct_change(12, fill_method=None)
    capital_df["commercial_credit_yoy"] = capital_df["commercial_credit"].pct_change(12, fill_method=None)
    capital_df["bank_reserves_yoy"] = capital_df["bank_reserves"].pct_change(12, fill_method=None)
    # Drop raw levels where YoY is the intended signal
    capital_df = capital_df.drop(columns=["bank_deposits", "commercial_credit", "bank_reserves"])
    capital_df.index.name = "date"

    return capital_df





#
# RUNNING THE CODE:
#

#capital_availability_constraints = get_capital_constraints(FRED_api_key)
#print('L0: Capital Availability Constraints Module. Gathering Data...')
#capital_availability_constraints.to_csv('capital_availability_constraints.csv')
#print('L0: Capital Availability Constraits Module Complete.')