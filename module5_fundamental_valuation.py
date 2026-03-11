import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
logger = logging.getLogger(__name__)


def run_gordon_growth_valuation(
        factors,
        raw,
        mapping_start_year,
        smoothing_months,
        enable_graph           = False,
):
    """
    Maps DFM factor scores to economic rate units via OLS regression,
    then plugs into Gordon Growth Model to estimate SPX fair value.

    Both EPS and denominator are smoothed by the same smoothing_months
    window, following Shiller CAPE methodology. Overvaluation expressed
    as raw %, historical percentile, and z-score (primary signal).
    All computed within the trimmed window only.

    Parameters
    ----------
    factors            : pd.DataFrame — factor scores from EM algorithm
    raw                : pd.DataFrame — unstandardized raw econ data
    mapping_start_year : int          — year to start ALL data:
                                        OLS calibration, fair value
                                        history, and signal ranking
    smoothing_months   : int          — rolling window in months applied
                                        to both EPS and denominator
                                        (e.g. 60 = 5yr, 120 = 10yr)
    enable_graph       : bool         — if True, renders and saves chart
    """

    mapping_start_date = f'{mapping_start_year}-01-01'

    # --- Align to common date range then trim to start year ---
    df      = factors.join(raw, how='inner')
    df      = df[df.index >= mapping_start_date]
    factors = factors[factors.index >= mapping_start_date]

    logger.debug(f'Data window:         {mapping_start_year} → present ({len(df)} months)')
    logger.debug(f'Smoothing window:    {smoothing_months} months ({smoothing_months/12:.0f} years) '
          f'— applied to EPS and denominator')

    # =========================================================
    # STEP 1: Map each factor to its economic rate anchor
    # OLS calibration on trimmed window only
    # =========================================================

    mappings     = {}
    mapped_rates = {}

    anchor_map = {
        'Discount'     : ('L0_treasury_10y',     'r_f', '10yr Treasury yield (%)'),
        'Risk_Premium' : ('L0_hy_credit_spread', 'erp', 'HY credit spread (%)'),
        'Growth'       : ('L4_gdp_yoy',          'g',   'GDP YoY growth rate (decimal)'),
    }

    logger.debug('=' * 65)
    logger.debug('Factor → Valuation Input Mappings (OLS)')
    logger.debug('=' * 65)

    # Loop through the three factor-to-rate pairs: Discount→treasury, Risk_Premium→HY spread, Growth→GDP.
    for factor_col, (raw_col, rate_name, description) in anchor_map.items():

        # Grab the historical record of both the factor score and the real-world rate side by side, drop any months where either is missing.
        pair = df[[factor_col, raw_col]].dropna()
        # Set up the regression. X is the factor score (the input), y is the real-world rate (the target). 
        # For example X = Discount factor scores from 1970-2025, y = actual 10yr Treasury yields from 1970-2025.
        X    = sm.add_constant(pair[[factor_col]]) 
        y    = pair[raw_col]
        # Run OLS to find the best-fit line: treasury_yield = α + β × Discount_factor. Store the fitted model.
        model                   = sm.OLS(y, X).fit()
        mappings[rate_name]     = model
        # Now apply that fitted line to the entire factor score history to generate a predicted rate for every month. 
        # This is the conversion — takes a unitless factor score like -2.79 and translates it into something like 4.2% treasury yield.
        X_full                  = sm.add_constant(factors[[factor_col]])
        mapped_rates[rate_name] = model.predict(X_full)

        logger.debug(f'\n  {factor_col} → {description}')
        logger.debug(f'    R²:      {model.rsquared:.3f}')
        logger.debug(f'    α:       {model.params["const"]:.4f}')
        logger.debug(f'    β:       {model.params[factor_col]:.4f}')
        logger.debug(f'    p(β):    {model.pvalues[factor_col]:.4f}')

    # =========================================================
    # STEP 2: Build raw denominator then smooth it
    # =========================================================

    r_f          = pd.Series(mapped_rates['r_f'], index=factors.index) / 100
    erp          = pd.Series(mapped_rates['erp'], index=factors.index) / 100
    g            = pd.Series(mapped_rates['g'],   index=factors.index)
    denom_raw    = r_f + erp - g
    denom_smooth = denom_raw.rolling(
                       window      = smoothing_months,
                       min_periods = smoothing_months
                   ).mean()
    denom_clipped = denom_smooth.clip(lower=0.02)

    # =========================================================
    # STEP 3: Get EPS and smooth it
    # EPS = SPX price / PE ratio
    # Smoothed EPS removes single-quarter earnings distortions
    # (COVID earnings collapse, one-time write-downs etc.)
    # =========================================================

    if 'L5_sp500_pe' in df.columns and 'L5_sp500_log' in df.columns:
        spx_price   = np.exp(df['L5_sp500_log'])
        eps_raw     = spx_price / df['L5_sp500_pe']
        eps_smooth  = eps_raw.rolling(
                          window      = smoothing_months,
                          min_periods = smoothing_months // 2
                      ).mean()
    else:
        raise ValueError('L5_sp500_pe or L5_sp500_log not found in raw data.')

    eps_raw_aligned    = eps_raw.reindex(factors.index)
    eps_smooth_aligned = eps_smooth.reindex(factors.index)

    # =========================================================
    # STEP 4: Compute fair value using smoothed EPS + smoothed denom
    # =========================================================

    fair_value = eps_smooth_aligned / denom_clipped

    # =========================================================
    # STEP 5: Compute over/undervaluation — three expressions
    #
    # raw %:      (actual / fair - 1) — regime-dependent, for reference
    # percentile: rank within trimmed window — bounded 0-100
    # z-score:    (raw - rolling_mean) / rolling_std
    #             PRIMARY SIGNAL — self-normalizing, regime-adjusting
    #             +2.0 = 2 std devs expensive vs own rolling history
    #             comparable across rate regimes
    # =========================================================

    spx_actual    = np.exp(df['L5_sp500_log']).reindex(factors.index)
    over_under    = (spx_actual / fair_value - 1)

    # percentile — within trimmed window
    over_under_pctile = over_under.rank(pct=True)

    # z-score — rolling, so early period has less history
    # min_periods=3 ensures at least 3 months before computing
    zscore_roll_mean  = over_under.rolling(window=smoothing_months, min_periods=3).mean()
    zscore_roll_std   = over_under.rolling(window=smoothing_months, min_periods=3).std()
    over_under_zscore = (over_under - zscore_roll_mean) / zscore_roll_std

    # =========================================================
    # STEP 6: Build output history dataframe
    # =========================================================

    df_history = pd.DataFrame({
        'fair_value'          : fair_value,
        'spx_actual'          : spx_actual,
        'over_under_pct'      : over_under,
        'over_under_pctile'   : over_under_pctile,
        'over_under_zscore'   : over_under_zscore,
        'r_f_mapped'          : r_f,
        'erp_mapped'          : erp,
        'g_mapped'            : g,
        'denom_raw'           : denom_raw,
        'denom_smooth'        : denom_smooth,
        'denom_clipped'       : denom_clipped,
        'eps_raw'             : eps_raw_aligned,
        'eps_smooth'          : eps_smooth_aligned,
    }, index=factors.index)

    latest      = df_history.dropna().iloc[-1]
    latest_date = df_history.dropna().index[-1]

    logger.info('=' * 65)
    logger.info(f'Gordon Growth Valuation (as of {latest_date.strftime("%Y-%m")})')
    logger.info(f'Data from {mapping_start_year}  |  {smoothing_months}m smoothing on EPS + denominator')
    logger.info('=' * 65)
    logger.info(f'  Mapped inputs:')
    logger.info(f'    Risk-free rate (r_f):        {latest.r_f_mapped*100:>7.2f}%')
    logger.info(f'    Equity risk premium (ERP):   {latest.erp_mapped*100:>7.2f}%')
    logger.info(f'    Growth rate (g):             {latest.g_mapped*100:>7.2f}%')
    logger.info(f'    Denominator raw:             {latest.denom_raw*100:>7.2f}%')
    logger.info(f'    Denominator smoothed:        {latest.denom_smooth*100:>7.2f}%')
    logger.info(f'    Denominator floored:         {latest.denom_clipped*100:>7.2f}%')
    logger.info(f'    EPS raw:                     ${latest.eps_raw:>7.2f}')
    logger.info(f'    EPS smoothed:                ${latest.eps_smooth:>7.2f}')
    logger.info('')
    logger.info(f'  SPX Fair Value:                ${latest.fair_value:>8.0f}')
    logger.info(f'  SPX Actual:                   ${latest.spx_actual:>8.0f}')
    logger.info(f'  Over/Undervaluation (raw):     {latest.over_under_pct:>+7.1%}')
    logger.info(f'  Historical Percentile:         {latest.over_under_pctile:>7.1%}  '
          f'(vs {mapping_start_year}–present)')
    logger.info(f'  Z-Score (primary signal):      {latest.over_under_zscore:>+7.2f}σ  '
          f'(+2.0 = expensive, -2.0 = cheap)')

    df_history.to_csv('gordon_growth_valuation.csv')
    logger.debug('')
    logger.debug('Valuation history saved to gordon_growth_valuation.csv')

    # =========================================================
    # STEP 7: Plot (optional)
    # =========================================================

    if enable_graph:

        plot_df = df_history[['spx_actual', 'fair_value',
                               'over_under_pct', 'over_under_pctile',
                               'over_under_zscore']].dropna()

        X_fit = sm.add_constant(plot_df['fair_value'])
        fit   = sm.OLS(plot_df['spx_actual'], X_fit).fit()
        trend = fit.predict(X_fit)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
        fig.patch.set_facecolor('#0f0f0f')
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor('#0f0f0f')
            ax.tick_params(colors='#aaaaaa')
            ax.xaxis.label.set_color('#aaaaaa')
            ax.yaxis.label.set_color('#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        # --- panel 1: price levels ---
        ax1.plot(plot_df.index, plot_df['spx_actual'], color='#00d4ff',
                 linewidth=1.5, label='SPX Actual')
        ax1.plot(plot_df.index, plot_df['fair_value'], color='#ff6b35',
                 linewidth=1.5, label=f'Fair Value ({smoothing_months}m smoothed)',
                 linestyle='--')
        ax1.plot(plot_df.index, trend, color='#ffcc00',
                 linewidth=1.0, label='Best Fit of Fair Value', linestyle=':')
        ax1.set_ylabel('Price Level ($)', color='#aaaaaa')
        ax1.set_title(
            f'SPX Actual vs Gordon Growth Fair Value  |  '
            f'Data from {mapping_start_year}  |  {smoothing_months}m smoothing',
            color='#ffffff', fontsize=13, pad=12)
        ax1.legend(facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc', fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # --- panel 2: z-score (primary signal) ---
        colors2 = ['#ff4444' if v > 0 else '#44ff88' for v in plot_df['over_under_zscore']]
        ax2.bar(plot_df.index, plot_df['over_under_zscore'],
                color=colors2, width=20, alpha=0.7)
        ax2.axhline( 0,   color='#555555', linewidth=0.8, linestyle='--')
        ax2.axhline( 2.0, color='#ff4444', linewidth=0.6, linestyle=':', alpha=0.6)
        ax2.axhline(-2.0, color='#44ff88', linewidth=0.6, linestyle=':', alpha=0.6)
        ax2.axhline( 1.0, color='#ff8888', linewidth=0.4, linestyle=':', alpha=0.4)
        ax2.axhline(-1.0, color='#88ff88', linewidth=0.4, linestyle=':', alpha=0.4)
        ax2.set_ylabel('Z-Score (σ)', color='#aaaaaa')
        ax2.set_title(
            'Overvaluation Z-Score  (primary signal)  '
            '|  +2σ = expensive  |  -2σ = cheap',
            color='#ffffff', fontsize=11, pad=8)
        ax2.annotate(
            f'Current: {latest.over_under_zscore:+.2f}σ',
            xy=(latest_date, latest.over_under_zscore),
            xytext=(10, -20), textcoords='offset points',
            color='#ff4444' if latest.over_under_zscore > 0 else '#44ff88',
            fontsize=9, arrowprops=dict(arrowstyle='->', color='#666666'),
        )

        # --- panel 3: historical percentile (secondary) ---
        colors3 = ['#ff4444' if v > 0.5 else '#44ff88'
                   for v in plot_df['over_under_pctile']]
        ax3.bar(plot_df.index, plot_df['over_under_pctile'] * 100,
                color=colors3, width=20, alpha=0.7)
        ax3.axhline(50, color='#555555', linewidth=0.8, linestyle='--')
        ax3.axhline(80, color='#ff4444', linewidth=0.6, linestyle=':', alpha=0.5)
        ax3.axhline(20, color='#44ff88', linewidth=0.6, linestyle=':', alpha=0.5)
        ax3.set_ylabel('Percentile', color='#aaaaaa')
        ax3.set_title(
            f'Historical Percentile Rank  (vs {mapping_start_year}–present)  '
            f'|  80th = expensive  |  20th = cheap',
            color='#ffffff', fontsize=11, pad=8)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}th'))
        ax3.set_ylim(0, 100)
        ax3.annotate(
            f'Current: {latest.over_under_pctile:.0%}',
            xy=(latest_date, latest.over_under_pctile * 100),
            xytext=(10, -20), textcoords='offset points',
            color='#ff4444' if latest.over_under_pctile > 0.5 else '#44ff88',
            fontsize=9, arrowprops=dict(arrowstyle='->', color='#666666'),
        )

        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right',
                 color='#aaaaaa')

        plt.tight_layout()
        plt.savefig('gordon_growth_valuation.png', dpi=150,
                    bbox_inches='tight', facecolor='#0f0f0f')
        plt.show()
        logger.debug('Chart saved to gordon_growth_valuation.png')

    return {
        'fair_value'          : latest.fair_value,
        'current_spx'         : latest.spx_actual,
        'over_under_pct'      : latest.over_under_pct,
        'over_under_pctile'   : latest.over_under_pctile,
        'over_under_zscore'   : latest.over_under_zscore,
        'mappings'            : mappings,
        'df_history'          : df_history,
    }





#
# RUNNING THE CODE
#

#factors = pd.read_csv('factor_scores.csv', index_col='date', parse_dates=True)
#raw     = pd.read_csv('all_econ_data.csv', index_col='date', parse_dates=True)

#valuation_results = run_gordon_growth_valuation(
    factors            = factors,
    raw                = raw,
    mapping_start_year = 1990,
    smoothing_months   = 12,     # 12 = 1yr, 120 = 10yr
    enable_graph       = False,
#)