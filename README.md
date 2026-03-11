# WELCOME:
-------------
# Setup is meant to be EASY for the non tech savvy. 
# Just copy and paste the mac / windows terminal commands below.

# D0NATIONS:
-------------
# Donations are always welcome if this tool was useful to you, but I don't want money to ever be a barrier to access.
# So this tool is 100% free, open source, and yours to do with what you will. Donation addresses below:

# BTC : bc1q2crr3jj0gpwrx68v9gqfv7pw34fc70qu073syl
# ETH : 0x399f9A262B0443e3B3B17B630543bdB4eB5651b7
# SOL : 4EoGxpqTpSRoEgooJQ213aUjq9Fzdnc3KeYzMYYDF591
# SUI : 0x775b86638b96033c568787eca49a7d2810bf7b18a6fbc7e0e03690ab1af3bb8d
# Base : 0x399f9A262B0443e3B3B17B630543bdB4eB5651b7
# Polygon : 0x399f9A262B0443e3B3B17B630543bdB4eB5651b7

# SETUP:
-------------
**Requirements:** Python 3.14, [uv](https://docs.astral.sh/uv/)

Download Python v3.14 here: https://www.python.org/downloads/

**1. Install UV (Copy/Paste this into terminal)**

Mac users:

curl -LsSf https://astral.sh/uv/install.sh | sh

Windows users:

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

**2. Navigate to where you want the model to be saved - Desktop recommended (Copy/Paste this into terminal)**

Mac users:

cd ~/Desktop

Windows users:

cd $HOME\Desktop

**3. Create project-econ folder and download the repo there (Copy/Paste this into terminal)**

Same for Mac and Windows:

git clone https://github.com/yourusername/project-econ.git

cd project-econ

**4. Install dependencies (Copy/Paste this into terminal)**

Same for Mac and Windows:

uv sync

**5. Add your API keys**
1. Go into the API_keys.py file
2. Follow the links and register for free API keys for economic data
3. Replace this line: '_replace_with_your_API_key_here_' with your API key for each platform. Keep your API key in 'quotes'. Save.

**6. Adjust parameters to your liking**

The main script is: econ_model.py

Open and modify econ_model.py to adjust the parameters to your liking, then save. Some recommended configurations are #commented.

**7. Run (Copy/Paste this into terminal)**

Same for Mac and Windows:

uv run econ_model.py

Note: The model prints progress to terminal as it runs. Total runtime is 
approximately 25-35 minutes depending on your machine and API response times.

# PHILOSOPHY BEHIND THE MODEL: 
------------------------------

A standard regression assumes the relationship between macroeconomic variables 
and target variable (in this case SPX) is fixed — i.e. the same beta on GDP growth 
or credit spreads in 1990 as in 2008 as in 2024. That assumption is empirically 
wrong. The sensitivity of markets to interest rates, labor conditions, and credit 
stress shifts dramatically across monetary regimes, business cycles, and structural 
breaks.

The deeper problem is that macroeconomic causality is not linear. A 
deterministic A→B→C framework — i.e. rising employment causes rising consumption 
causes rising earnings causes rising equity prices — ignores that the same 
input produces different outputs depending on where you are in the cycle. 
Rising employment in an overheating economy with an inverted yield curve is 
not the same signal as rising employment in an early recovery. The direction 
is the same. The implication is not. A model that cannot distinguish between 
those two states will be wrong at exactly the moments that matter most.

The correct framework is therefore a dynamic factor model with state-space 
regression — using a walk-forward Kalman filter to produce genuinely 
out-of-sample predictions, time-varying betas, and empirically calibrated 
probability distributions. In practice this works as follows:

This model treats data relationships as time-varying. The EM algorithm 
compresses ~165 macroeconomic series into three latent factors — Growth, 
Discount, and Risk Premium — that are never directly observable but can be 
inferred from the cross-section of data. The Kalman filter then estimates the 
relationship between those factors and target variable returns, allowing the 
betas to evolve as regimes change. The final estimate is out-of-sample: 
the model only uses information available at that point in time.

The result is a framework that adapts — one that was not told what 2008 looked 
like before predicting 2009, and was not told what 2020 looked like before 
predicting 2021. That is the standard a useful macro model should be held to.

Note that Steps 4 and 5 produce a full-sample regression and fundamental 
valuation estimate respectively — this is why Growth, Discount, and Risk 
Premium were chosen as the three factors, as they map directly to the inputs 
of a Gordon Growth Model. The final prediction in Steps 6–10 is driven by the 
walk-forward Kalman optimization, and leverages quintiling and time-varying 
regression coefficients to produce a calibrated out-of-sample probability 
distribution over the forward return window.

# ROADMAP: 
------------------------------

Coming in V2: In combination with the statistical model, a secondary A->B->C 
deterministic model which may be less accurate at short term predictions,
but which may complement existing results with a more manually definied
and human interpreted regime classification.

-------------

**8. Errors**
Often the back-end of FRED or EIA is down for hours at a time, which throws a 504 error. Rather than build a retry block, the script will break with a traceback error.
This is free and I don't update it too often, so any issues just toss the code / terminal output in to your preferred AI model to debug.

**9. Disclaimers**
PLEASE READ THIS DISCLAIMER IN FULL BEFORE USING THIS SOFTWARE.
BY DOWNLOADING, INSTALLING, OR RUNNING THIS CODE, YOU ACKNOWLEDGE
THAT YOU HAVE READ, UNDERSTOOD, AND AGREED TO ALL TERMS BELOW.

─────────────────────────────────────────────────────────────────

NOT FINANCIAL ADVICE

This software, model, code, output, documentation, and any associated 
commentary (collectively, the "Software") is provided strictly for 
educational, informational, and research purposes only. Nothing contained 
in this repository constitutes, or should be construed as, financial advice, 
investment advice, trading advice, securities recommendations, or any other 
form of professional financial, legal, tax, or regulatory guidance of any 
kind.

The author of this Software is not acting in any registered or compensated 
advisory capacity in connection with this Software. This Software is provided 
free of charge, makes no personalized recommendations to any individual, and 
does not constitute "investment advice" as defined under the Investment Advisers 
Act of 1940 or any applicable federal or state securities law. Any views, outputs, 
or signals produced by this Software do not represent and should not be attributed 
to any employer, regulator, or affiliated institution, past, present, or future. 
This Software is and will remain an independent, personal, open-source project.

This Software does not constitute an offer to sell, a solicitation of an offer
to buy, or a recommendation of any security, financial instrument, or investment 
strategy. Nothing herein should be interpreted as forming a client-adviser 
relationship of any kind between the author and any user of this Software.
Use of this Software does not create any fiduciary duty or obligation 
on the part of the author toward any user.

─────────────────────────────────────────────────────────────────

NO GUARANTEE OF ACCURACY — DATA AND CALCULATIONS

The outputs of this model — including but not limited to predicted returns, 
factor scores, quintile rankings, valuation estimates, probability estimates, 
directional signals, and any other numerical or qualitative output — may be 
materially inaccurate, incomplete, or misleading. Inaccuracies may arise from, 
but are not limited to:

  - Errors, revisions, or omissions in third-party source data (FRED, BLS, 
    BEA, EIA, Shiller/Yale, Yahoo Finance, or any other data provider)
  - API outages, data feed interruptions, or stale data
  - Numerical errors, floating point precision limits, or bugs in the 
    underlying code or its dependencies
  - Model misspecification, overfitting, or structural breaks in the 
    relationships the model was estimated on
  - Publication lag adjustments that may not perfectly reflect real-time 
    data availability in all historical periods
  - Use of revised rather than real-time vintage data, which may cause 
    backtest results to appear stronger than they would have been in practice
  - Changes in statistical relationships over time that invalidate 
    historically estimated parameters
  - Software dependency updates that alter numerical behavior

The author makes no representation or warranty — express or implied — as to 
the accuracy, completeness, reliability, fitness for purpose, or timeliness 
of any output produced by this Software. All calculations should be 
independently verified before being relied upon for any purpose.

─────────────────────────────────────────────────────────────────

PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE RESULTS

All backtest results, out-of-sample statistics, R² values, directional 
accuracy figures, quintile hit rates, and any other historical performance 
metrics presented by this Software reflect model performance over a specific 
historical period and under specific market conditions. Results:

  - Are produced with the benefit of hindsight over a defined historical window
  - May reflect survivorship bias, data mining, or overfitting to historical 
    patterns that do not generalize
  - Are not a guarantee, prediction, or reliable indicator of future model 
    performance or future market returns
  - May deteriorate significantly in future market environments, regime 
    changes, or periods not represented in the training data

There is no assurance that the model will achieve similar results in the 
future. Markets involve substantial risk of loss, including the 
possible loss of the entire amount invested. 

─────────────────────────────────────────────────────────────────

ASSUMPTION OF RISK / LIMITATION OF LIABILITY

Any investment, trading, or financial decision made in reliance on this 
Software, in whole or in part, is made entirely at your own risk. The author 
expressly disclaims all liability for any direct, indirect, incidental, 
special, consequential, punitive, or exemplary damages of any kind — 
including but not limited to financial losses, lost profits, lost data, 
business interruption, or any other pecuniary or non-pecuniary loss — 
arising out of or in connection with your use of, or inability to use, 
this Software, regardless of whether such damages were foreseeable and 
regardless of whether the author was advised of the possibility of such 
damages.

This limitation of liability applies to the fullest extent permitted by 
applicable law.

─────────────────────────────────────────────────────────────────

THIRD-PARTY DATA AND SERVICES

This Software relies on data and services provided by third parties, 
including the Federal Reserve Bank of St. Louis (FRED), the U.S. Bureau 
of Labor Statistics (BLS), the U.S. Bureau of Economic Analysis (BEA), 
the U.S. Energy Information Administration (EIA), Robert Shiller / Yale 
University, and Yahoo Finance. The author has no affiliation with, and 
makes no representation regarding, any of these data providers. The 
author accepts no responsibility for the accuracy, availability, or 
continuity of any third-party data or service. API availability is 
entirely outside the author's control.

─────────────────────────────────────────────────────────────────

DONATIONS

Acceptance of donations does not create any obligation, warranty, support 
commitment, or ongoing relationship of any kind between the author and the donor.

─────────────────────────────────────────────────────────────────

NO GUARANTEE OF MAINTENANCE

The author is under no obligation to maintain, update, fix, or continue 
development of this Software at any time. This is a personal project provided 
freely and without any service commitment.

─────────────────────────────────────────────────────────────────

NO WARRANTY

THIS SOFTWARE IS PROVIDED "AS IS" AND "AS AVAILABLE", WITHOUT WARRANTY 
OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, AND NON-INFRINGEMENT. THE ENTIRE RISK AS TO THE QUALITY AND 
PERFORMANCE OF THE SOFTWARE IS WITH YOU.

─────────────────────────────────────────────────────────────────

By using this Software you confirm that you have read, understood, and 
agreed to all of the above, and that you are using this Software solely 
for your own educational and informational purposes. If you do not agree 
to these terms, do not use this Software.