import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key

# 4 Core technology/productivity variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Technological efficiency:
# 1. MFPNFBS total_factor_productivity      pure tech efficiency, independent of labor/capital      expressed as index value, annual forward-filled

# Capital and capacity:
# 2. PNFI nonres_fixed_investment_yoy     real private nonresidential fixed investment YoY          expressed as decimal. 5% = 0.05
# 3. TCU capacity_utilization               % of productive capacity currently in use               expressed as decimal. 75% = 0.75

# Technology infrastructure:
# 4. Y033RC1Q027SBEA it_investment_yoy      investment in computing, software, IT infra YoY         expressed as decimal. 5% = 0.05, quarterly forward-filled


def get_technology_constraints(FRED_api_key):
    fred = Fred(FRED_api_key)
    # Annual series → forward-fill to monthly
    tfp = fred.get_series("MFPNFBS").resample("MS").ffill()
    # Quarterly series → forward-fill to monthly
    nonres_investment = fred.get_series("PNFI").resample("MS").ffill()
    it_investment = fred.get_series("Y033RC1Q027SBEA").resample("MS").ffill()
    # Monthly series
    tech_df = pd.DataFrame({
        "total_factor_productivity": tfp,
        "nonres_fixed_investment":   nonres_investment,
        "capacity_utilization":      fred.get_series("TCU"),
        "it_investment":             it_investment,
    })
    # convert to decimal format
    tech_df["capacity_utilization"] = tech_df["capacity_utilization"] / 100
    # YoY transformations — drop raw levels where YoY is the intended signal
    tech_df["nonres_fixed_investment_yoy"] = tech_df["nonres_fixed_investment"].pct_change(12, fill_method=None)
    tech_df["it_investment_yoy"] = tech_df["it_investment"].pct_change(12, fill_method=None)
    tech_df = tech_df.drop(columns=["nonres_fixed_investment", "it_investment"])
    tech_df.index.name = "date"

    return tech_df





#
# RUNNING THE CODE:
#

#technology_constraints = get_technology_constraints(FRED_api_key)
#print('L0: Running Technology and Productivity Constraints Module. Gathering Data...')
#technology_constraints.to_csv('tech_and_productivity_constraints.csv')
#print('L0: Technology and Productivity Constraints Module Complete.')