import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from module1_data_standardize import standardize_data
import logging
logger = logging.getLogger(__name__)


def init_discount_factor_value(df_std, DISCOUNT_COLS):

    # --- Subset to discount bucket ---
    df_discount = df_std[DISCOUNT_COLS]

    # --- Drop columns that are entirely NaN ---
    df_discount = df_discount.dropna(axis=1, how='all')
    valid_cols  = df_discount.columns.tolist()
    dropped     = set(DISCOUNT_COLS) - set(valid_cols)
    if dropped:
        logger.warning(f'Dropped all-NaN columns: {dropped}')

    # --- Impute NaNs with column mean (0 since standardized) for PCA ---
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_discount)

    # --- Run PCA ---
    pca = PCA(n_components=1)
    pca.fit(X)

    # --- Extract PC1 as initial discount factor time series ---
    discount_factor_series = pca.transform(X)[:, 0]
    discount_factor_series = pd.Series(discount_factor_series, index=df_discount.index, name='pc1_discount')

    # --- Diagnostics ---
    loadings           = pd.Series(pca.components_[0], index=valid_cols, name='pc1_loading').sort_values(ascending=False)
    variance_explained = pca.explained_variance_ratio_[0]

    logger.debug('Discount Factor PCA Initialization')
    logger.debug(f'  Series in bucket:   {len(DISCOUNT_COLS)} ({len(valid_cols)} after dropping all-NaN)')
    logger.debug(f'  Variance explained: {variance_explained:.1%}')
    logger.debug(f'  Initial value (F0): {float(discount_factor_series.iloc[0]):.4f}')
    logger.debug('  Top 10 positive loadings:\n' + loadings.head(10).round(3).to_string())
    logger.debug('  Top 10 negative loadings:\n' + loadings.tail(10).round(3).to_string())

    # --- Sign correction: discount factor ↑ = higher rates / tighter policy ---
    # Anchor: L0_treasury_10y must correlate positively with discount factor
    anchor_corr = discount_factor_series.corr(df_std['L0_treasury_10y'].reindex(discount_factor_series.index))
    if anchor_corr < 0:
        discount_factor_series = discount_factor_series * -1
        logger.debug('  Sign corrected — pc1_discount inverted to match L0_treasury_10y')

    return float(discount_factor_series.iloc[0]), discount_factor_series