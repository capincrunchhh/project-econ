import pandas as pd
from fredapi import Fred
from API_keys import FRED_api_key

# 5 Core labor variables. ALL DATA MONTHLY AS MOST ECONOMIC MODELS FUNCTION MONTHLY

# Labor supply constraint:
# 1. labor_force_participation   % of working-age population in labor force   expressed as decimal. 62.5% = 0.625

# Labor utilization:
# 2. payroll_yoy                 YoY growth in total nonfarm payrolls         expressed as decimal. 5% = 0.05
# 3. avg_weekly_hours            average weekly hours per employee            expressed as hours (e.g. 34.5 = 34.5 hrs)

# Labor capacity and cost:
# 4. labor_productivity          output per hour worked, nonfarm business     expressed as index value (2012=100), quarterly forward-filled
# 5. unit_labor_cost             labor cost per unit of output, nonfarm biz   expressed as index value (2012=100), quarterly forward-filled


def get_labor_constraints(FRED_api_key):
    fred = Fred(FRED_api_key)
    # Monthly series
    labor_df = pd.DataFrame({
        "labor_force_participation": fred.get_series("CIVPART"),
        "payroll_employment": fred.get_series("PAYEMS"),
        "avg_weekly_hours": fred.get_series("AWHAETP"),
    })
    # YoY transformation for payrolls — drop raw level - this can be noisy in model if not normalized to %
    labor_df["payroll_yoy"] = labor_df["payroll_employment"].pct_change(12)
    labor_df = labor_df.drop(columns=["payroll_employment"])
    # LFP published as e.g. 62.5 → normalize to decimal
    labor_df["labor_force_participation"] = labor_df["labor_force_participation"] / 100
    # Quarterly series → forward-fill to monthly
    productivity = fred.get_series("OPHNFB").resample("MS").ffill()
    unit_labor_cost = fred.get_series("ULCNFB").resample("MS").ffill()
    labor_df["labor_productivity"] = productivity
    labor_df["unit_labor_cost"] = unit_labor_cost
    labor_df.index.name = "date"
    labor_df = labor_df.sort_index()

    return labor_df




#
# RUNNING THE CODE:
#

#labor_constraints = get_labor_constraints(FRED_api_key)
#print('L0: Running Labor Constraints Module. Gathering Data...')
#labor_constraints.to_csv('labor_constraints.csv')
#print('L0: Labor Constraints Module Complete.')





#Why unemployment rate and wage growth belong more in Layer 3 / 4: This is subtle but extremely important. The key distinction is:
#Layer 0 = structural constraints (capacity limits). Layer 3/4 = realized economic outcomes (how the system is currently behaving)
#Unemployment rate and wage growth are primarily outcome variables, not constraint variables.

#Why YoY is superior to trailing averages for macro constraint measurement: It captures structural labor expansion or contraction.
#The key issue is seasonality and structural interpretability. Trailing averages answer a different question.
#Problem with trailing averages: seasonality distortion, Employment is highly seasonal.
#Retail hires massively in November–December. Construction rises in summer. Education employment follows school cycles.
#nstitutional macro models overwhelmingly use YoY