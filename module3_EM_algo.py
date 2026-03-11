import pandas as pd
import numpy as np
from scipy.linalg import solve
import logging
logger = logging.getLogger(__name__)

from module1_data_standardize import standardize_data
from module2_factor_growth import init_growth_factor_value
from module2_factor_discount import init_discount_factor_value
from module2_factor_risk_premium import init_risk_premium_factor_value
from module2_data_build_f0_and_lambda_df import build_lambda_df_init, build_F0


def enforce_positive_definite(M, jitter=1e-8):
    """
    Ensure a matrix is symmetric positive definite.
    Symmetrizes first, then clips any negative eigenvalues to jitter.
    Called after updating P_smooth and Q to prevent covariance matrices
    from going non-positive-definite due to numerical accumulation errors.
    """
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, jitter)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def run_em_dfm(Y, lambda_init, F0, n_iter=500, tol=1e-6):
    """
    EM (Expectation-Maximization) algorithm for Dynamic Factor Model.

    E-step: given current parameters, estimate hidden factor scores at every t
            using the Kalman filter (forward) and RTS smoother (backward)
    M-step: given those factor score estimates, solve analytically for the
            parameters that maximize the likelihood of the observed data
    -------------------------------------------------------------------------
    State space form:
        Observation: Y(t) = Lambda @ F(t) + eps(t)
        Transition:  F(t) = A @ F(t-1) + eta(t)
    -------------------------------------------------------------------------
    Variables:
        Y(t)      — observed data at time t (N series)
        F(t)      — latent factor scores at time t [Growth, Discount, RiskPrem]
        Lambda    — loading matrix: how much each series moves per unit of factor
        A         — transition matrix: how factors at t-1 predict factors at t
        eps(t)    — observation noise at time t, drawn from N(0, R)
        eta(t)    — state noise at time t, drawn from N(0, Q)
        R         — observation noise covariance (N x N diagonal)
                    each diagonal element = idiosyncratic variance of that series
        Q         — state noise covariance (3 x 3)
                    how much randomness enters the factor evolution each period
        F0        — initial factor state at t=0 [Growth, Discount, RiskPrem]
        P0        — initial state covariance (3 x 3)
                    uncertainty about where the factors start
        T         — number of time periods (434 months)
        N         — number of observed series (137)
    -------------------------------------------------------------------------
    Parameters
    ----------
    Y           : np.ndarray (T x N) — observed data, NaNs allowed
    lambda_init : np.ndarray (N x 3) — initial loading matrix from PCA/correlation
    F0          : list (3,)          — initial factor state vector [growth, discount, risk_prem]
    n_iter      : int                — maximum number of EM iterations
    tol         : float              — convergence threshold on log-likelihood change
    -------------------------------------------------------------------------
    Returns
    -------
    results : dict
        F_smooth   — (T x 3)   smoothed factor scores at every time step
        Lambda     — (N x 3)   estimated loading matrix after EM convergence
        A          — (3 x 3)   estimated factor transition matrix
        R          — (N x N)   estimated observation noise covariance (diagonal)
        Q          — (3 x 3)   estimated state noise covariance
        F0         — (3,)      updated initial state after EM
        P0         — (3 x 3)   updated initial covariance after EM
        ll_history — list      log-likelihood value at each iteration
    """
    logger.info('=' * 65)
    logger.info('Running EM algorithm...')

    # Unpack dimensions from Y
    # Shape describes the size of an array: (T, N) means T rows and N columns
    # T = 434 months (rows), N = 137 series (columns)
    T, N = Y.shape
    n_factors = lambda_init.shape[1]

    # --- Initialize parameters ---
    Lambda = lambda_init.copy()          # shape (137, 3) — starting loading matrix from PCA/correlation init
    A      = np.eye(n_factors) * 0.9    # shape (3, 3) — transition matrix: factors assumed 90% persistent month-to-month
    Q      = np.eye(n_factors) * 0.1    # shape (3, 3) — state noise: random shock entering each factor per month
    R      = np.eye(N) * 0.1            # shape (137, 137) — observation noise: idiosyncratic variance per series
                                         # R is diagonal — each series has its own independent noise on the diagonal,
                                         # zeros everywhere else, meaning series noise doesn't bleed into other series
                                         # Q is factor-level randomness (3x3), R is series-level randomness (137x137)
    F0     = np.array(F0, dtype=float)  # shape (3,) — starting factor values [Growth, Discount, RiskPrem] at t=0
    P0     = np.eye(n_factors)          # shape (3, 3) — starting uncertainty about where the factors begin

    # Tracks log-likelihood after each EM iteration to monitor convergence
    # Should increase (become less negative) each iteration until it flattens out
    ll_history = []

    # --- Pre-compute observation masks ---
    # NaN positions don't change between iterations — no need to recompute each time
    # obs_masks[t]        — for each month t, which of the 137 series have real data (not NaN)
    #                       used in the Kalman filter forward pass to skip missing series
    # series_obs_masks[i] — for each series i, which of the 434 months have real data (not NaN)
    #                       used in the M-step to only update Lambda/R from observed months
    obs_masks        = [~np.isnan(Y[t])    for t in range(T)]
    series_obs_masks = [~np.isnan(Y[:, i]) for i in range(N)]

    for iteration in range(n_iter):

        # ===================================================
        # E-STEP: Kalman filter + smoother
        # Computes E[F(t)] and E[F(t)F(t)'] given parameters
        # ===================================================

        # --- Forward pass: Kalman filter ---
        # Create empty arrays of the right shape before filling them month by month in the loop
        # Shape (T, n_factors) = 434 rows × 3 columns — one row per month, one column per factor
        # Shape (T, n_factors, n_factors) = 434 slices of 3×3 matrices — one 3×3 uncertainty
        #   matrix per month. 3×3 (not just 3 numbers) because the three factors are correlated
        #   so their uncertainty must be tracked jointly, not independently
        F_filt = np.zeros((T, n_factors))            # shape (434, 3)   — filtered factor means
        P_filt = np.zeros((T, n_factors, n_factors)) # shape (434, 3, 3) — filtered factor covariances
        F_pred = np.zeros((T, n_factors))            # shape (434, 3)   — predicted factor means
        P_pred = np.zeros((T, n_factors, n_factors)) # shape (434, 3, 3) — predicted factor covariances
        ll     = 0.0                                  # log-likelihood accumulator for this iteration

        for t in range(T):

            # --- Predict ---
            # Before looking at data for month t, predict where factors should be
            # based on where they were last month and the transition matrix A
            # @ is matrix multiplication (dot product), not element-wise multiplication
            # A @ F0 means: multiply transition matrix A by factor vector F0 to get predicted factors
            # t=0: use F0/P0 as starting condition; all later months use prior filtered estimate
            if t == 0:
                F_pred[t] = A @ F0           # predicted factors = A × initial factor state
                P_pred[t] = A @ P0 @ A.T + Q # predicted uncertainty = A × initial uncertainty × A' + state noise
            else:
                F_pred[t] = A @ F_filt[t-1]           # predicted factors = A × last month's filtered estimate
                P_pred[t] = A @ P_filt[t-1] @ A.T + Q # predicted uncertainty propagated forward + state noise

            # --- Use pre-computed mask ---
            # Identify which of the 137 series actually have data this month (not NaN)
            # If no series have data at all this month, skip update and carry prediction forward
            obs_mask = obs_masks[t]
            if obs_mask.sum() == 0:
                F_filt[t] = F_pred[t]
                P_filt[t] = P_pred[t]
                continue

            # Subset Lambda, R, and Y to only the series observed this month
            # Lambda_obs: shape (n_obs, 3) — loadings for observed series only
            # R_obs: shape (n_obs, n_obs) — R is 137×137 but we only need noise values for
            #        observed series. R is diagonal so we pull out the relevant diagonal values
            #        and rebuild a smaller diagonal matrix — "give me noise for series I can see"
            # y_obs: shape (n_obs,) — actual data values for observed series this month
            Lambda_obs = Lambda[obs_mask]
            R_obs      = np.diag(np.diag(R)[obs_mask])
            y_obs      = Y[t, obs_mask]

            # --- Innovation ---
            # The innovation is the surprise: actual data minus what the model predicted
            # y_hat: what we expected to observe given our predicted factors (@ = matrix multiply)
            # innov: the gap between what actually happened and what we predicted
            # S: total uncertainty in that gap — from both factor uncertainty and observation noise
            #    shape (n_obs, n_obs) — uncertainty matrix across all observed series this month
            y_hat = Lambda_obs @ F_pred[t]
            innov = y_obs - y_hat
            S     = Lambda_obs @ P_pred[t] @ Lambda_obs.T + R_obs

            # --- Ridge-stabilized inversion ---
            # S needs to be inverted to compute the Kalman gain
            # Sometimes floating point errors make S nearly zero in some directions,
            # causing division by near-zero which produces garbage or crashes
            # Adding 1e-8 to the diagonal puts a tiny floor under every value —
            # the numerical equivalent of adding a small constant to a denominator
            # to prevent divide-by-zero. np.linalg.solve is more stable than np.linalg.inv
            S_stable = S + np.eye(S.shape[0]) * 1e-8
            S_inv    = np.linalg.solve(S_stable, np.eye(S.shape[0]))

            # --- Kalman gain ---
            # K controls how much to trust the data vs the model prediction
            # High K = trust data more, update factor estimate strongly
            # Low K = trust model more, update factor estimate weakly
            # K is large when prediction uncertainty (P_pred) is high relative to observation noise (S)
            # shape (3, n_obs) — maps from observed series space back to factor space
            K = P_pred[t] @ Lambda_obs.T @ S_inv

            # --- Update ---
            # Revise the factor estimate by adding a fraction of the surprise (innovation)
            # @ is matrix multiplication: K @ innov multiplies gain matrix (3×n_obs) by
            #   surprise vector (n_obs×1) to get a 3×1 correction, then adds to prediction
            # F_filt: best estimate of factors after seeing this month's data
            # P_filt: uncertainty shrinks after incorporating the data observation
            F_filt[t] = F_pred[t] + K @ innov
            P_filt[t] = (np.eye(n_factors) - K @ Lambda_obs) @ P_pred[t]

            # --- Log-likelihood contribution ---
            # Measures how probable this month's data was under the current model parameters
            # slogdet computes log(det(S)) in a numerically stable way
            # Accumulates across all 434 months to give total log-likelihood for this iteration
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                ll += -0.5 * (obs_mask.sum() * np.log(2 * np.pi) + logdet +
                              innov @ S_inv @ innov)

        ll_history.append(ll)

        # --- Backward pass: RTS smoother ---
        # The Kalman filter only looks backward — at month 50 it only knows months 1-50
        # The smoother revises every month's estimate using the full 434-month picture
        # Like the difference between a journalist writing in real time (filter) vs a
        # historian writing 30 years later who can look back at 1995 knowing what came after
        #
        # Create empty arrays of the right shape before filling them in the backward loop:
        # F_smooth shape (434, 3)    — one revised factor score per month per factor
        # P_smooth shape (434, 3, 3) — one 3×3 uncertainty matrix per month (3×3 not just 3
        #                              numbers because the three factors' uncertainties are correlated)
        # G        shape (434, 3, 3) — smoother gain: controls how much future info revises
        #                              each month's estimate, operates on full 3×3 factor covariance
        F_smooth = np.zeros((T, n_factors))
        P_smooth = np.zeros((T, n_factors, n_factors))
        G        = np.zeros((T, n_factors, n_factors))

        # Initialize smoother at the last month with the filtered values
        # The last month has no future data to incorporate so smoothed = filtered
        F_smooth[-1] = F_filt[-1]
        P_smooth[-1] = enforce_positive_definite(P_filt[-1])

        # Work backward from second-to-last month to the first
        # At each step, revise the factor estimate using information from the future
        # G[t]: how much to adjust the current filtered estimate based on future information
        # F_smooth[t]: filtered estimate + correction based on how next month's smoothed
        #              estimate differs from what we predicted next month would be
        # P_smooth[t]: revised uncertainty after incorporating future information
        for t in range(T-2, -1, -1):
            P_pred_stable = P_pred[t+1] + np.eye(n_factors) * 1e-8  # ridge stabilization
            G[t]          = P_filt[t] @ A.T @ np.linalg.solve(P_pred_stable, np.eye(n_factors))
            F_smooth[t]   = F_filt[t] + G[t] @ (F_smooth[t+1] - F_pred[t+1])
            P_smooth[t]   = enforce_positive_definite(
                P_filt[t] + G[t] @ (P_smooth[t+1] - P_pred[t+1]) @ G[t].T
            )

        # --- Cross-covariance for M-step ---
        # P_cross[t] = E[F(t) × F(t-1)'] — expected joint covariance of adjacent factor states
        # shape (434, 3, 3) — one 3×3 matrix per month
        # Needed by M-step to estimate how consecutive factor states relate to each other,
        # which determines the transition matrix A
        P_cross = np.zeros((T, n_factors, n_factors))
        for t in range(1, T):
            P_cross[t] = P_smooth[t] @ G[t-1].T + \
                         np.outer(F_smooth[t], F_smooth[t-1])

        # ===================================================
        # M-STEP: Update parameters via closed-form MLE
        # ===================================================

        # --- Update A ---
        # A is the transition matrix: F(t) = A × F(t-1) + noise
        # S1 accumulates E[F(t) × F(t-1)'] — how factors at t relate to factors at t-1
        # S2 accumulates E[F(t-1) × F(t-1)'] — variance of lagged factors
        # A = S1 / S2 — equivalent to OLS regression of F(t) on F(t-1) across all months
        S1 = np.zeros((n_factors, n_factors))
        S2 = np.zeros((n_factors, n_factors))
        for t in range(1, T):
            S1 += P_cross[t]
            S2 += P_smooth[t-1] + np.outer(F_smooth[t-1], F_smooth[t-1])
        A = S1 @ np.linalg.solve(S2 + np.eye(n_factors) * 1e-8, np.eye(n_factors))

        # --- Update Q ---
        # Q is the state noise covariance: how much random shock enters factors each month
        # Computed as average unexplained variance in factor transitions after accounting for A
        # enforce_positive_definite ensures Q remains a valid covariance matrix
        Q = np.zeros((n_factors, n_factors))
        for t in range(1, T):
            EFFt  = P_smooth[t]   + np.outer(F_smooth[t],   F_smooth[t])
            EFFt1 = P_smooth[t-1] + np.outer(F_smooth[t-1], F_smooth[t-1])
            Q += EFFt - A @ P_cross[t].T - P_cross[t] @ A.T + A @ EFFt1 @ A.T
        Q = enforce_positive_definite(Q / (T - 1))

        # --- Update Lambda and R ---
        # For each of the 137 series, re-estimate its loadings on the 3 factors
        # and its idiosyncratic noise variance — only using months where it was observed
        for i in range(N):
            obs_t = series_obs_masks[i]  # which months does series i have data
            if obs_t.sum() == 0:
                continue

            # F_obs: shape (n_obs, 3) — smoothed factor scores at months where series i was observed
            # y_i:   shape (n_obs,)   — actual values of series i at those same months
            F_obs = F_smooth[obs_t]
            y_i   = Y[obs_t, i]

            # obs_indices: integer positions of observed months, needed for indexing P_smooth
            # EFF: shape (3, 3) — E[F × F'] summed over observed months
            #      the "X'X" matrix in OLS terms — factor self-covariance used to solve for loadings
            obs_indices = np.where(obs_t)[0]
            EFF = sum(P_smooth[t] + np.outer(F_smooth[t], F_smooth[t])
                      for t in obs_indices)

            # Lambda[i]: new loading vector for series i — OLS of y_i on smoothed factors
            # solve(EFF, F_obs.T @ y_i) is the matrix equivalent of (X'X)^-1 X'y in OLS
            # @ is matrix multiplication: F_obs.T (3×n_obs) @ y_i (n_obs×1) = (3×1) vector
            Lambda[i] = np.linalg.solve(EFF + np.eye(n_factors) * 1e-8, F_obs.T @ y_i)

            # resid: what's left of series i after removing the factor-explained component
            # R[i,i]: idiosyncratic variance = average squared residual + uncertainty correction
            #         the Lambda @ P_smooth @ Lambda term corrects for the fact that factors
            #         are estimated with uncertainty, not observed directly
            resid   = y_i - F_obs @ Lambda[i]
            R[i, i] = (resid @ resid + Lambda[i] @
                       sum(P_smooth[t] for t in obs_indices) @
                       Lambda[i]) / obs_t.sum()

        # --- Update initial conditions ---
        # After seeing all 434 months, revise beliefs about where factors started
        # F0 becomes the smoothed estimate at t=0 — uses all data, not just early months
        # P0 becomes the smoothed uncertainty at t=0
        F0 = F_smooth[0]
        P0 = enforce_positive_definite(P_smooth[0])

        # --- Convergence check ---
        # Stop if relative improvement in log-likelihood falls below threshold
        # Relative (not absolute) so it scales correctly regardless of ll magnitude
        # tol=1e-6 means stop when improvement is less than 0.0001% of current value
        if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) / abs(ll_history[-2]) < tol:
            logger.info(f'  Converged at iteration {iteration+1}')
            break

        # Print progress every 50 iterations to monitor the run
        if (iteration + 1) % 50 == 0:
            logger.info(f'  Iteration {iteration+1:4d} / {n_iter}  |  log-likelihood: {ll:.2f}')

    return {
        'F_smooth':   F_smooth,
        'Lambda':     Lambda,
        'A':          A,
        'R':          R,
        'Q':          Q,
        'F0':         F0,
        'P0':         P0,
        'll_history': ll_history,
    }





#
# RUNNING THE CODE
#

# --- Load and standardize ---
#df = pd.read_csv('all_econ_data.csv', index_col='date', parse_dates=True)
#df_std, TARGET_VARIABLE, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS, CURRENTLY_UNUSED_COLS = standardize_data(df)

# --- Build initialization inputs ---
#growth_init,    pca_growth       = init_growth_factor_value(df_std, GROWTH_COLS, verbose=False)
#discount_init,  pca_discount     = init_discount_factor_value(df_std, DISCOUNT_COLS, verbose=False)
#risk_prem_init, pca_risk_premium = init_risk_premium_factor_value(df_std, RISK_PREMIUM_COLS, verbose=False)

#F0        = build_F0(growth_init, discount_init, risk_prem_init)
#lambda_df = build_lambda_df_init(df_std, GROWTH_COLS, DISCOUNT_COLS, RISK_PREMIUM_COLS,
#                                 pca_growth, pca_discount, pca_risk_premium)

# --- Prepare Y matrix (T x N) — active buckets only, NaNs preserved ---
#all_cols = GROWTH_COLS + DISCOUNT_COLS + RISK_PREMIUM_COLS
#Y        = df_std[all_cols].values.astype(float)

# --- Run EM with PCA informed initialization ---
#print('Running EM algorithm...')
#print(f'  Data shape:   {Y.shape}')
#print(f'  F0:           {[round(v, 3) for v in F0]}')
#print(f'  Lambda shape: {lambda_df.values.shape}')
#print()

#results = run_em_dfm(
#    Y           = Y,
#    lambda_init = lambda_df.values,
#    F0          = F0,
#    n_iter      = 500,
#    tol         = 1e-6,
#)

# --- MULTI-SEED RANDOM INITIALIZATION TEST — commented out, for diagnostics only ---
# Run EM from 20 random starting points to find statistically optimal solution
# Note: random seeds may find higher likelihood but lose economic interpretability
# PCA informed initialization trades ~3.8% likelihood for interpretable factors
# Uncomment to re-run if bucket structure changes materially
#print('Running multi-seed random initialization test...')
#print(f'  Data shape:   {Y.shape}')
#print(f'  Lambda shape: {lambda_df.values.shape}')
#print()
# best_ll     = -np.inf
# best_result = None
# best_seed   = None
# for seed in range(20):
#     np.random.seed(seed)
#     F0_test     = list(np.random.randn(3))
#     lambda_test = pd.DataFrame(
#                     np.random.randn(*lambda_df.shape) * 0.1,
#                     index=lambda_df.index, columns=lambda_df.columns)
#     print(f'  Seed {seed:2d} running...', end=' ', flush=True)
#     result = run_em_dfm(Y=Y, lambda_init=lambda_test.values,
#                         F0=F0_test, n_iter=500, tol=1e-6)
#     final_ll = result['ll_history'][-1]
#     is_best  = final_ll > best_ll
#     print(f'll={final_ll:.2f}{"  ← new best" if is_best else ""}')
#     if is_best:
#         best_ll     = final_ll
#         best_result = result
#         best_seed   = seed
#print()
#print(f'Best seed:                       {best_seed}')
#print(f'Best log-likelihood (random):    {best_ll:.2f}')
#print(f'PCA informed initialization:     -46114.71')
#print(f'Difference:                      {best_ll - (-46114.71):.2f}')

# --- Save outputs ---
#dates    = df_std.index
#F_smooth = pd.DataFrame(results['F_smooth'], index=dates,
#                        columns=['Growth', 'Discount', 'Risk_Premium'])
#Lambda   = pd.DataFrame(results['Lambda'],   index=all_cols,
#                        columns=['Growth', 'Discount', 'Risk_Premium'])

#F_smooth.to_csv('factor_scores.csv')
#Lambda.to_csv('lambda_estimated.csv')

#print()
#print(f'Final log-likelihood: {results["ll_history"][-1]:.2f}')
#print(f'Factor scores saved to factor_scores.csv')
#print(f'Estimated lambda saved to lambda_estimated.csv')