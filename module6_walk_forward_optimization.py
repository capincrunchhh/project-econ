import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)


'''
Important note on data leakage:
-------------------------------
This is designed to consume results of walk-forward EM algo as inputs. Those results are free of data-leakage, therefore this is free of data-leakage. 
'''


def run_kalman_regression(
        F_smooth,
        df_raw,
        REGRESSION_TARGET,
        forward_months,
        start_year     = 1970, #defaults to 1970 if unspecified
        beta_drift_q   = 0.001,
        obs_noise_r    = None,
        in_sample_r2   = None,
        enable_graph   = False,
):
    """
    Time-varying parameter (TVP) Kalman filter regression for Step 6.

    SPX returns are demeaned before fitting so factors explain deviations
    from the unconditional mean return, not the mean itself. The mean is
    added back to every prediction at the end.

    No intercept term — 3 parameters only: beta_growth, beta_discount, beta_riskprem.
    Includes bias analysis and quintile ranking to diagnose whether direction
    signal is monotonic even when magnitude prediction is poor.

    Parameters
    ----------
    F_smooth          : pd.DataFrame — factor scores from Step 3 EM algorithm
    df_raw            : pd.DataFrame — raw unstandardized data
    REGRESSION_TARGET : str          — e.g. 'L5_sp500_tr_3m'
    forward_months    : int          — prediction horizon in months
    start_year        : int          — restrict to this year onwards
                                       removes pre-modern regime data
    beta_drift_q      : float        — state noise controlling beta drift speed
                                       0.0001 = near-static betas
                                       0.001  = moderate drift (recommended)
                                       0.01   = fast-adapting betas
    obs_noise_r       : float|None   — observation noise variance
                                       None = estimated from SPX return variance
    in_sample_r2      : float|None   — Step 4 R² for comparison
    enable_graph      : bool         — render 4-panel output chart
    """

# =========================================================
    # SETUP
    # =========================================================
    # Takes inputs and prepares data for the filter.
    # Trims everything to start_year, builds the forward SPX return
    # target by shifting the return series forward by forward_months,
    # then aligns factor scores with SPX returns on matching dates
    # so both arrays are exactly the same length before the filter runs.
    #
    # Factor columns are dynamic — supports 3-factor baseline and any
    # n-factor extension (e.g. 4-factor from Step 9a composite test).

    start_date = pd.Timestamp(f'{start_year}-01-01')
    F_use      = F_smooth[F_smooth.index >= start_date].copy()
    df_use     = df_raw[df_raw.index >= start_date].copy()

    spx            = df_use[[REGRESSION_TARGET]].copy()
    spx['spx_fwd'] = spx[REGRESSION_TARGET].shift(-forward_months)
    spx_aligned    = spx[['spx_fwd']]

    df_aligned = F_use.join(spx_aligned, how='inner')
    dates      = df_aligned.index
    T          = len(dates)

    # Observation matrix X: factor scores only, no intercept
    # shape (T, n_params) — one row per month per factor
    # dynamic — picks up whatever columns are in F_smooth (3 on first pass, or 4+ in later steps)
    factor_cols = [c for c in F_smooth.columns]
    n_params    = len(factor_cols)
    X           = np.column_stack([df_aligned[c].values for c in factor_cols])

    y_raw = df_aligned['spx_fwd'].values   # raw forward SPX returns, shape (T,)

    # Demean SPX returns: remove the unconditional positive drift of SPX
    # so factors only explain variation around that mean, not the mean itself.
    # The mean gets added back to every prediction at the end.
    obs_mask = ~np.isnan(y_raw)
    y_mean   = float(np.nanmean(y_raw[obs_mask]))
    y        = y_raw - y_mean   # demeaned target fed into the Kalman filter

    # =========================================================
    # INITIALIZE
    # =========================================================
    # Sets starting conditions for the Kalman filter before it sees any data.
    #
    # Q:       controls how fast betas are allowed to drift each month.
    #          small Q = betas barely move, large Q = betas very reactive.
    #          this replaces the arbitrary train_months window of rolling OLS —
    #          instead of choosing a window, you choose how fast betas can evolve.
    #
    # R:       how noisy SPX returns are around the model's prediction.
    #          estimated from the variance of actual demeaned SPX returns.
    #
    # burn_in: runs a quick no-intercept OLS on the first 24 months to give
    #          the filter a sensible starting point for betas rather than
    #          starting from zero and converging cold through early predictions.
    #
    # P_0:     initial uncertainty about the starting beta values.
    #          set large (1.0) so the filter updates aggressively in the
    #          early months before settling into a stable rhythm.

    Q = np.eye(n_params) * beta_drift_q

    if obs_noise_r is None:
        R = float(np.nanvar(y[obs_mask]))
    else:
        R = float(obs_noise_r)

    burn_in   = min(24, T // 4)
    burn_mask = obs_mask[:burn_in]
    if burn_mask.sum() >= n_params + 1:
            from numpy.linalg import lstsq

            X_burn = X[:burn_in][burn_mask]
            y_burn = y[:burn_in][burn_mask]

            # drop any rows with NaN in X
            valid = np.isfinite(X_burn).all(axis=1)
            logger.debug(f"burn-in: {burn_mask.sum()} obs, {valid.sum()} valid after NaN drop")

            if valid.sum() >= n_params + 1:
                beta_0, _, _, _ = lstsq(X_burn[valid], y_burn[valid], rcond=None)
            else:
                logger.warning("Insufficient valid burn-in rows after NaN drop — using zero init")
                beta_0 = np.zeros(n_params)

    P_0 = np.eye(n_params) * 1.0

    logger.info('=' * 65)
    logger.info('Walk-Forward Optimization: Kalman Regression (Time-Varying Parameters)')
    logger.info('=' * 65)
    logger.info(f'  Start year:          {start_year}')
    logger.info(f'  Prediction horizon:  {forward_months} month(s)')
    logger.info(f'  Beta drift Q:        {beta_drift_q}')
    logger.info(f'  Observation noise R: {R:.6f}')
    logger.info(f'  SPX mean return:     {y_mean:+.4f}  (added back to all predictions)')
    logger.info(f'  Intercept:           none — factors explain deviations from mean only')
    logger.info(f'  Burn-in obs:         {burn_mask.sum()} months')
    logger.info(f'  Total months:        {T}')
    logger.info(f'  Date range:          {dates[0].strftime("%Y-%m")} → {dates[-1].strftime("%Y-%m")}')

    # =========================================================
    # KALMAN FILTER FORWARD PASS
    # =========================================================
    # The core loop — processes every month sequentially from start to end.
    # At each month t it does three things:
    #
    # 1. PREDICT:
    #    Assumes betas this month equal last month's betas (random walk assumption).
    #    Uncertainty about the betas grows by Q because they may have drifted
    #    since last month — the longer since last update, the less certain we are.
    #
    # 2. STORE AND PREDICT SPX:
    #    Uses predicted betas × today's factor scores to generate a forward
    #    SPX prediction. Adds y_mean back to convert from demeaned to actual
    #    return space. Critically, this uses only information available up to
    #    month t-1 — making every prediction genuinely out-of-sample.
    #
    # 3. UPDATE:
    #    Observes what SPX actually returned this month.
    #    Computes the innovation (surprise = actual - predicted).
    #    Kalman gain K decides how much weight to give new data vs prior betas:
    #      high K = new data more reliable than prior, update betas aggressively
    #      low K  = prior betas more reliable than new data, update slowly
    #    Updates beta estimates and shrinks uncertainty after incorporating data.
    #    If SPX return is NaN (future months), skips update and carries
    #    current betas forward unchanged into next month.

    beta_hist  = np.zeros((T, n_params))        # beta estimates at every month
    P_hist     = np.zeros((T, n_params, n_params))  # uncertainty at every month
    pred_hist  = np.full(T, np.nan)             # SPX predictions at every month
    innov_hist = np.full(T, np.nan)             # innovations (surprises) at every month

    beta_t = beta_0.copy()
    P_t    = P_0.copy()

    for t in range(T):

        # --- Predict step ---
        beta_pred    = beta_t.copy()   # predicted beta = last beta (random walk)
        P_pred       = P_t + Q         # uncertainty expands by state noise Q

        # Store prediction before update — uses only info up to t-1
        beta_hist[t] = beta_pred
        P_hist[t]    = P_pred
        pred_hist[t] = float(X[t] @ beta_pred) + y_mean   # add mean back

        # --- Update step ---
        if np.isnan(y[t]):
            # No realized return available — future date or missing data
            # carry betas forward unchanged, no learning this month
            beta_t = beta_pred
            P_t    = P_pred
            continue

        innov         = y[t] - float(X[t] @ beta_pred)   # surprise this month
        innov_hist[t] = innov

        S      = float(X[t] @ P_pred @ X[t]) + R   # total prediction uncertainty
        K      = P_pred @ X[t] / S                  # Kalman gain
        beta_t = beta_pred + K * innov               # updated betas
        P_t    = (np.eye(n_params) - np.outer(K, X[t])) @ P_pred  # updated uncertainty

    # =========================================================
    # BUILD OUTPUT DATAFRAMES
    # =========================================================
    # Packages raw arrays from the filter loop into clean pandas
    # dataframes with proper date indices for analysis and export.
    #
    # df_betas:       beta estimates at every month — shows how Growth,
    #                 Discount, Risk_Premium betas evolved over time.
    #                 P_trace = sum of diagonal of uncertainty matrix,
    #                 a scalar measure of total beta uncertainty each month.
    #
    # df_predictions: predicted vs realized return at every month,
    #                 plus the innovation (surprise) at each step.
    #                 realized_return uses raw (not demeaned) returns
    #                 so comparison is in interpretable return space.

    df_betas = pd.DataFrame(
        beta_hist,
        index   = dates,
        columns = [f'beta_{c.lower()}' for c in factor_cols],
    )
    df_betas['P_trace'] = [np.trace(P_hist[t]) for t in range(T)]

    df_predictions = pd.DataFrame({
        'predicted_return' : pred_hist,
        'realized_return'  : y_raw,
        'innovation'       : innov_hist,
    }, index=dates)

    # Restrict evaluation to post burn-in, non-NaN months only
    eval_start = dates[burn_in]
    df_eval    = df_predictions[df_predictions.index >= eval_start].dropna(
                     subset=['realized_return', 'predicted_return'])

    # =========================================================
    # CORE METRICS
    # =========================================================
    # Standard model evaluation computed on raw (uncorrected) predictions.
    # These metrics are the clean OOS test — no bias correction applied here.
    #
    # OOS R²:          does the model beat the naive benchmark of predicting
    #                  the historical mean return every single month?
    #                  negative = model is worse than that simple benchmark
    #                  on squared error, even though direction may be right
    #
    # Directional acc: what % of individual months did the model correctly
    #                  predict whether SPX went up or down?
    #                  evaluated independently for each of the 405 months
    #
    # MAE:             mean absolute error. average absolute prediction error in return 
    #                  % terms more intuitive than squared error for sizing context

    ss_res = ((df_eval['realized_return'] - df_eval['predicted_return']) ** 2).sum()
    ss_tot = ((df_eval['realized_return'] - df_eval['realized_return'].mean()) ** 2).sum()
    oos_r2 = 1 - ss_res / ss_tot

    correct_direction = (
        np.sign(df_eval['predicted_return']) == np.sign(df_eval['realized_return'])
    )
    directional_acc = correct_direction.mean()
    mae = (df_eval['realized_return'] - df_eval['predicted_return']).abs().mean()

    # =========================================================
    # BIAS ANALYSIS
    # =========================================================
    # Diagnoses the systematic error structure of the model across
    # all 405 out-of-sample predictions.
    #
    # Mean/median error:    is the model consistently too high or too low
    #                       on average across all months?
    #                       computed as predicted minus realized so:
    #                       positive = model overpredicts on average
    #                       negative = model underpredicts on average
    #
    # Scale factor:         compares spread of predictions vs realized returns.
    #                       realized_std / predicted_std
    #                       > 1 = predictions too compressed, model underconfident
    #                       < 1 = predictions too spread, model overconfident
    #                       your result (0.90x) = predictions slightly more
    #                       dispersed than reality — mild overconfidence
    #
    # Conditional bias:     splits into months where model was bullish vs bearish
    #                       separately to test if error is symmetric across regimes.
    #                       symmetric bias = simple calibration problem
    #                       asymmetric bias = regime-dependent structural issue
    #
    # Bias corrected pred:  applies mean_bias and scale_factor to current prediction.
    #                       this is calibration not lookahead — historical evaluation
    #                       metrics always use raw uncorrected predictions.
    #                       full-sample correction acts like regularization:
    #                       averages error structure across all regimes including
    #                       hostile ones, trading regime-specific precision for
    #                       cross-regime generalization — analogous to L2 shrinkage.

    errors      = df_eval['predicted_return'] - df_eval['realized_return']
    mean_bias   = errors.mean()
    median_bias = errors.median()
    bias_direction = 'overpredicts' if mean_bias > 0 else 'underpredicts'

    pred_std     = df_eval['predicted_return'].std()
    realized_std = df_eval['realized_return'].std()
    scale_factor = realized_std / pred_std if pred_std > 0 else np.nan

    df_eval = df_eval.copy()
    df_eval['pred_bias_corrected'] = (
        (df_eval['predicted_return'] - mean_bias) * scale_factor
    )

    ss_res_bc  = ((df_eval['realized_return'] - df_eval['pred_bias_corrected']) ** 2).sum()
    oos_r2_bc  = 1 - ss_res_bc / ss_tot

    correct_dir_bc = (
        np.sign(df_eval['pred_bias_corrected']) == np.sign(df_eval['realized_return'])
    )
    dir_acc_bc = correct_dir_bc.mean()
    mae_bc     = (df_eval['realized_return'] - df_eval['pred_bias_corrected']).abs().mean()

    bullish_mask = df_eval['predicted_return'] > y_mean
    bearish_mask = df_eval['predicted_return'] <= y_mean
    bullish_bias = (df_eval.loc[bullish_mask, 'predicted_return'] -
                    df_eval.loc[bullish_mask, 'realized_return']).mean()
    bearish_bias = (df_eval.loc[bearish_mask, 'predicted_return'] -
                    df_eval.loc[bearish_mask, 'realized_return']).mean()

    # =========================================================
    # QUINTILE RANKING ANALYSIS
    # =========================================================
    # The most important diagnostic section.
    #
    # Splits all 405 predictions into 5 equal buckets ranked from most
    # bearish (Q1) to most bullish (Q5) by predicted return magnitude.
    # These are NOT consecutive months in time — each quintile contains
    # 81 months scattered across 34 years of history that happened to
    # have similar predicted return levels.
    #
    # For each bucket, asks: across the average of all observations in
    # that quintile (individual monthly errors net out, leaving only the
    # average accuracy of the signal at that confidence level), what did
    # SPX actually return in months where the model was this bullish or bearish?
    #
    # If the model has real signal, realized returns should increase
    # monotonically — meaning both consistently and in order of magnitude —
    # from Q1 to Q5. Errors within each bucket cancel out when averaged,
    # so what remains is the pure directional signal strength at each level.
    #
    # Monotonicity score: what % of adjacent quintile pairs are in the
    # correct order? A pair here means the average predicted return for
    # a quintile vs the average realized return for that same quintile —
    # one representative pair per bucket, not month-by-month comparisons.
    # Individual errors within each bucket net out, so the pair reflects
    # the model's aggregate accuracy at that predicted confidence level.
    #
    #   100% = every quintile step up in prediction corresponds to a
    #           step up in average realized return — perfectly ranked
    #   75%  = 3 of 4 adjacent pairs in correct order
    #   50%  = random — no better than chance at ranking regimes
    #
    # This is the definitive test of whether the directional signal is
    # rankable and actionable across regimes, not just accidentally right
    # on individual months. A model can have negative OOS R² (poor magnitude)
    # and still have 100% monotonicity (perfect ranking) — direction signal
    # and magnitude signal are independent properties.

    try:
        df_eval['pred_quintile'] = pd.qcut(
            df_eval['predicted_return'], 5,
            labels=['Q1 (bearish)', 'Q2', 'Q3', 'Q4', 'Q5 (bullish)']
        )
        quintile_stats = df_eval.groupby('pred_quintile', observed=True).agg(
            mean_realized   = ('realized_return', 'mean'),
            median_realized = ('realized_return', 'median'),
            count           = ('realized_return', 'count'),
            pct_positive    = ('realized_return', lambda x: (x > 0).mean()),
        )

        means         = quintile_stats['mean_realized'].values
        n_pairs       = len(means) - 1
        correct_pairs = sum(means[i] < means[i+1] for i in range(n_pairs))
        monotonicity  = correct_pairs / n_pairs

    except Exception as e:
        quintile_stats = None
        monotonicity   = np.nan
        logger.debug(f'  Quintile analysis failed: {e}')

    # =========================================================
    # CURRENT PREDICTION
    # =========================================================
    # Applies the model to the most recent month's factor scores.
    #
    # Raw prediction:       Kalman betas × latest factor scores + SPX mean
    #                       uses only information available today
    #
    # Bias corrected:       raw prediction adjusted using full-sample error
    #                       calibration (mean_bias + scale_factor)
    #                       this is the number used for Step 7 positioning
    #                       applying full-sample correction to a single forward
    #                       prediction is calibration not lookahead — analogous
    #                       to a weather model correcting tomorrow's forecast
    #                       using 30 years of measured forecast errors

    current_pred    = pred_hist[-1]
    current_pred_bc = (current_pred - mean_bias) * scale_factor
    current_betas = dict(zip(factor_cols, beta_hist[-1]))

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    logger.info('=' * 65)
    logger.info('Kalman Regression Results')
    logger.info('=' * 65)
    logger.info(f'  Evaluation period:           {eval_start.strftime("%Y-%m")} → {dates[-1].strftime("%Y-%m")}')
    logger.info(f'  Out-of-sample predictions:   {len(df_eval)}')
    logger.info('')
    logger.info('  --- Core Metrics (raw predictions, no correction) ---')
    logger.info(f'  OOS R²:                      {oos_r2:.4f}')
    if in_sample_r2 is not None:
        flag = '✓ PASS' if oos_r2 >= 0.10 else '✗ FAIL'
        logger.info(f'  In-sample R² (Step 4):      {in_sample_r2:.4f}')
        logger.info(f'  Gate (OOS R² ≥ 0.10):       {flag}')
    logger.info(f'  Directional accuracy:        {directional_acc:.1%}  (>55% = useful)')
    logger.info(f'  Mean Abs Error:                         {mae:.1%}')
    logger.info('')
    logger.debug('  --- Bias Analysis ---')
    logger.debug(f'  Mean error (pred - actual):  {mean_bias:+.4f}  → model {bias_direction} on average')
    logger.debug(f'  Median error:                {median_bias:+.4f}')
    logger.debug(f'  Predicted return std:        {pred_std:.4f}')
    logger.debug(f'  Realized return std:         {realized_std:.4f}')
    if scale_factor > 1:
        logger.debug(f'  Scale factor:                {scale_factor:.2f}x  (predictions too compressed)')
    else:
        logger.debug(f'  Scale factor:                {scale_factor:.2f}x  (predictions too spread by {1/scale_factor:.1f}x)')
    logger.debug('')
    logger.debug('  --- Conditional Bias ---')
    logger.debug(f'  Bullish months bias:         {bullish_bias:+.4f}  (positive = overpredicts when bullish)')
    logger.debug(f'  Bearish months bias:         {bearish_bias:+.4f}  (negative = underpredicts when bearish)')
    logger.debug('')
    logger.info('  --- After Bias Correction ---')
    logger.info(f'  OOS R² (corrected):          {oos_r2_bc:.4f}')
    logger.info(f'  Directional acc (corrected): {dir_acc_bc:.1%}')
    logger.info(f'  Mean Abs Error (corrected):             {mae_bc:.1%}')
    logger.debug('')
    logger.debug('  --- Quintile Ranking ---')
    logger.debug(f'  Monotonicity score:          {monotonicity:.0%}  '
          f'(100% = perfectly ranked, 50% = random)')
    if quintile_stats is not None:
        logger.debug(f'  {"Quintile":<20} {"Mean Realized":>14} {"% Positive":>11} {"Count":>6}')
        logger.debug('  ' + '-' * 55)
        for q, row in quintile_stats.iterrows():
            logger.debug(f'  {str(q):<20} {row["mean_realized"]:>+13.1%} '
                  f'{row["pct_positive"]:>10.1%} {int(row["count"]):>6}')
    logger.debug('')
    logger.debug('  --- Current Betas ---')
    for k, v in current_betas.items():
        logger.debug(f'    {k:<16} {v:+.4f}')
    logger.debug('')
    logger.debug(f'  Current {forward_months}m prediction (raw):       {current_pred:+.1%}')
    logger.debug(f'  Current {forward_months}m prediction (corrected): {current_pred_bc:+.1%}')
    logger.debug('=' * 65)

    # =========================================================
    # SAVE
    # =========================================================

    df_predictions['pred_bias_corrected'] = np.nan
    df_predictions.loc[df_eval.index, 'pred_bias_corrected'] = \
        df_eval['pred_bias_corrected'].values
    df_predictions.to_csv('kalman_predictions.csv')
    df_betas.to_csv('kalman_betas.csv')
    logger.debug('')
    logger.debug('Predictions saved to kalman_predictions.csv')
    logger.debug('Betas      saved to kalman_betas.csv')

    # =========================================================
    # OPTIONAL GRAPH
    # =========================================================
    # 4-panel chart:
    # Panel 1: predicted vs realized returns over time with bias corrected overlay
    # Panel 2: rolling directional accuracy — shows consistency of signal over time
    # Panel 3: time-varying betas — shows how factor relationships evolved
    # Panel 4: quintile bar chart — shows monotonic ranking of realized returns
    #          by predicted return bucket, the definitive signal quality test

    if enable_graph:

        fig, axes = plt.subplots(4, 1, figsize=(14, 17), sharex=False)
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes:
            ax.set_facecolor('#0f0f0f')
            ax.tick_params(colors='#aaaaaa')
            ax.xaxis.label.set_color('#aaaaaa')
            ax.yaxis.label.set_color('#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        ax1, ax2, ax3, ax4 = axes

        ax1.plot(df_eval.index, df_eval['realized_return'],
                 color='#00d4ff', linewidth=1.5, label='Realized Return')
        ax1.plot(df_eval.index, df_eval['predicted_return'],
                 color='#ff6b35', linewidth=1.5, linestyle='--',
                 label='Kalman Predicted (raw)')
        ax1.plot(df_eval.index, df_eval['pred_bias_corrected'],
                 color='#ffcc00', linewidth=1.0, linestyle=':',
                 label='Kalman Predicted (bias corrected)')
        ax1.axhline(0,      color='#555555', linewidth=0.8, linestyle='--')
        ax1.axhline(y_mean, color='#888888', linewidth=0.6, linestyle=':',
                    label=f'SPX mean ({y_mean:+.2%})')
        ax1.set_ylabel('Return', color='#aaaaaa')
        ax1.set_title(
            f'Kalman TVP  |  {forward_months}m  |  Q={beta_drift_q}  |  '
            f'start={start_year}  |  OOS R²={oos_r2:.3f}  |  '
            f'OOS R²(bc)={oos_r2_bc:.3f}',
            color='#ffffff', fontsize=11, pad=12)
        ax1.legend(facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc', fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0%}'))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))

        roll_window = min(24, max(6, len(df_eval) // 10))
        rolling_dir = correct_direction.rolling(roll_window).mean()
        ax2.plot(rolling_dir.index, rolling_dir * 100,
                 color='#ffcc00', linewidth=1.5,
                 label=f'Rolling {roll_window}-month directional accuracy')
        ax2.axhline(50, color='#555555', linewidth=0.8, linestyle='--',
                    label='Random (50%)')
        ax2.axhline(55, color='#44ff88', linewidth=0.6, linestyle=':',
                    alpha=0.6, label='Useful threshold (55%)')
        ax2.set_ylabel('Directional Accuracy (%)', color='#aaaaaa')
        ax2.set_title('Rolling Directional Accuracy',
                      color='#ffffff', fontsize=11, pad=8)
        ax2.legend(facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc', fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(5))

        ax3.plot(df_betas.index, df_betas['beta_growth'],
                 color='#00d4ff', linewidth=1.2, label='β Growth')
        ax3.plot(df_betas.index, df_betas['beta_discount'],
                 color='#ff6b35', linewidth=1.2, label='β Discount')
        ax3.plot(df_betas.index, df_betas['beta_riskprem'],
                 color='#44ff88', linewidth=1.2, label='β Risk_Premium')
        ax3.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
        ax3.set_ylabel('Beta', color='#aaaaaa')
        ax3.set_title('Time-Varying Betas (Kalman Filter)',
                      color='#ffffff', fontsize=11, pad=8)
        ax3.legend(facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc', fontsize=9)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator(5))

        if quintile_stats is not None:
            colors = ['#ff4444' if v < 0 else '#44ff88'
                      for v in quintile_stats['mean_realized']]
            bars = ax4.bar(
                range(len(quintile_stats)),
                quintile_stats['mean_realized'] * 100,
                color=colors, alpha=0.8, edgecolor='#333333'
            )
            ax4.set_xticks(range(len(quintile_stats)))
            ax4.set_xticklabels(quintile_stats.index, color='#aaaaaa', fontsize=9)
            ax4.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
            ax4.axhline(y_mean * 100, color='#888888', linewidth=0.6,
                        linestyle=':', label=f'Unconditional mean ({y_mean:+.2%})')
            ax4.set_ylabel('Mean Realized Return (%)', color='#aaaaaa')
            ax4.set_title(
                f'Quintile Ranking  |  Monotonicity={monotonicity:.0%}  |  '
                f'Q1=most bearish prediction, Q5=most bullish  |  '
                f'bars=average realized return across all months in each bucket',
                color='#ffffff', fontsize=10, pad=8)
            ax4.legend(facecolor='#1a1a1a', edgecolor='#333333',
                       labelcolor='#cccccc', fontsize=9)
            ax4.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:+.1f}%'))
            for bar, (_, row) in zip(bars, quintile_stats.iterrows()):
                ax4.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.1,
                         f'n={int(row["count"])}',
                         ha='center', va='bottom',
                         color='#aaaaaa', fontsize=8)

        plt.tight_layout()
        plt.savefig('kalman_validation.png', dpi=150,
                    bbox_inches='tight', facecolor='#0f0f0f')
        plt.show()
        logger.debug('Chart saved to kalman_validation.png')

    return {
        'oos_r2'          : oos_r2,
        'oos_r2_bc'       : oos_r2_bc,
        'directional_acc' : directional_acc,
        'dir_acc_bc'      : dir_acc_bc,
        'mae'             : mae,
        'mae_bc'          : mae_bc,
        'mean_bias'       : mean_bias,
        'scale_factor'    : scale_factor,
        'monotonicity'    : monotonicity,
        'quintile_stats'  : quintile_stats,
        'df_predictions'  : df_predictions,
        'df_betas'        : df_betas,
        'current_pred'    : current_pred,
        'current_pred_bc' : current_pred_bc,
        'current_betas'   : current_betas,
        'y_mean'          : y_mean,
    }





#
# STANDALONE TEST
#

#F_smooth  = pd.read_csv('factor_scores.csv', index_col='date', parse_dates=True)
#df_raw    = pd.read_csv('all_econ_data.csv', index_col='date', parse_dates=True)

#kalman_results = run_kalman_regression(
    F_smooth          = F_smooth,
    df_raw            = df_raw,
    REGRESSION_TARGET = 'L5_sp500_tr_3m',
    forward_months    = 3,
    start_year        = 1990, # 1970 start had a better result, but 1990 enables correction logic to work
    beta_drift_q      = 0.001,
    obs_noise_r       = None,
    in_sample_r2      = 0.1992,
    enable_graph      = False,
#)