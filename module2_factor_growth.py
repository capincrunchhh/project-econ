import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from module1_data_standardize import standardize_data
import logging
logger = logging.getLogger(__name__)


def init_growth_factor_value(df_std, GROWTH_COLS):

    # --- Subset to growth bucket ---
    df_growth = df_std[GROWTH_COLS]

    # --- Impute NaNs with column mean (0 since standardized) for PCA ---
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_growth)

    # --- Run PCA ---
    pca = PCA(n_components=1)
    pca.fit(X)

    # --- Extract PC1 as initial growth factor time series ---
    growth_factor_series = pca.transform(X)[:, 0]
    growth_factor_series = pd.Series(growth_factor_series, index=df_growth.index, name='pc1_growth')

    # --- Diagnostics ---
    loadings           = pd.Series(pca.components_[0], index=GROWTH_COLS, name='pc1_loading').sort_values(ascending=False)
    variance_explained = pca.explained_variance_ratio_[0]

    logger.debug(f'Growth Factor PCA Initialization')
    logger.debug(f'  Series in bucket:   {len(GROWTH_COLS)}')
    logger.debug(f'  Variance explained: {variance_explained:.1%}')
    logger.debug(f'  Initial value (F0): {float(growth_factor_series.iloc[0]):.4f}')
    logger.debug('  Top 10 positive loadings:\n' + loadings.head(10).round(3).to_string())
    logger.debug('  Top 10 negative loadings:\n' + loadings.tail(10).round(3).to_string())

    # --- Sign correction: growth factor ↑ = better growth ---
    # Anchor: L4_gdp_yoy must correlate positively with growth factor
    anchor_corr = growth_factor_series.corr(df_std['L4_gdp_yoy'].reindex(growth_factor_series.index))
    if anchor_corr < 0:
        growth_factor_series = growth_factor_series * -1
        logger.debug('  Sign corrected — pc1_growth inverted to match L4_gdp_yoy')

    return float(growth_factor_series.iloc[0]), growth_factor_series