import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from module1_data_standardize import standardize_data
import logging
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def init_risk_premium_factor_value(df_std, RISK_PREMIUM_COLS):

    # --- Subset to risk premium bucket ---
    df_risk_premium = df_std[RISK_PREMIUM_COLS]

    # --- Drop columns that are entirely NaN ---
    df_risk_premium = df_risk_premium.dropna(axis=1, how='all')
    valid_cols      = df_risk_premium.columns.tolist()
    dropped         = set(RISK_PREMIUM_COLS) - set(valid_cols)
    if dropped:
        logger.warning(f'Dropped all-NaN columns: {dropped}')

    # --- Impute NaNs with column mean (0 since standardized) for PCA ---
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_risk_premium)

    # --- Run PCA ---
    pca = PCA(n_components=1)
    pca.fit(X)

    # --- Extract PC1 as initial risk premium factor time series ---
    risk_premium_factor_series = pca.transform(X)[:, 0]
    risk_premium_factor_series = pd.Series(risk_premium_factor_series, index=df_risk_premium.index, name='pc1_risk_premium')

    # --- Diagnostics ---
    loadings           = pd.Series(pca.components_[0], index=valid_cols, name='pc1_loading').sort_values(ascending=False)
    variance_explained = pca.explained_variance_ratio_[0]

    logger.debug('Risk Premium Factor PCA Initialization')
    logger.debug(f'  Series in bucket:   {len(RISK_PREMIUM_COLS)} ({len(valid_cols)} after dropping all-NaN)')
    logger.debug(f'  Variance explained: {variance_explained:.1%}')
    logger.debug(f'  Initial value (F0): {float(risk_premium_factor_series.iloc[0]):.4f}')
    logger.debug('  Top 10 positive loadings:\n' + loadings.head(10).round(3).to_string())
    logger.debug('  Top 10 negative loadings:\n' + loadings.tail(10).round(3).to_string())

    # --- Sign correction: risk premium factor ↑ = more stress / fear ---
    # Anchor: L2_vix must correlate positively with risk premium factor
    anchor_corr = risk_premium_factor_series.corr(df_std['L2_vix'].reindex(risk_premium_factor_series.index))
    if anchor_corr < 0:
        risk_premium_factor_series = risk_premium_factor_series * -1
        logger.debug('  Sign corrected — pc1_risk_premium inverted to match L2_vix')

    return float(risk_premium_factor_series.iloc[0]), risk_premium_factor_series