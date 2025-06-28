"""
FeatureScaler

Feature scaling module for financial time series data.

Provides FeatureScaler, an automatic scaler that detects optimal scaling methods
for mixed financial features (e.g., RSI, ATR) using heuristics based on data distribution.

Scalers supported:
- MinMaxScaler for bounded features like RSI (0-100 range)
- RobustScaler for features with outliers
- StandardScaler as default fallback

Includes feature statistics tracking for MLOps monitoring and model interpretation.
"""

import numpy as np
import pandas as pd

from scipy.stats import iqr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Automatically detects and applies optimal scaling for financial time series features.
    Handles mixed-scale features like RSI (0-100) and ATR (volatility) without hardcoding.
    """

    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}

    def _auto_detect_scaler(self, data):
        """Heuristic-based scaler selection using financial data characteristics"""
        # Calculate distribution properties
        q1, q3 = np.percentile(data, [25, 75])
        iqr_val = q3 - q1
        range_val = np.max(data) - np.min(data)
        has_outliers = (iqr_val > 0) and (range_val / iqr_val > 4)

        # Check for common financial indicator ranges
        is_bounded = (
            (np.min(data) >= 0) and
            (np.max(data) <= 100) and
            (range_val >= 50)  # Typical for RSI-like features
        )

        if is_bounded:
            return MinMaxScaler(feature_range=(0, 1)), 'minmax'
        
        elif has_outliers:
            return RobustScaler(), 'robust'
        
        else:
            return StandardScaler(), 'standard'

    def fit(self, X, y=None):
        for col in X.columns:
            column_data = X[col].values.reshape(-1, 1)
            scaler, scaler_type = self._auto_detect_scaler(column_data)
            self.scalers[col] = scaler.fit(column_data)

            # Store metadata for MLOps monitoring
            self.feature_stats[col] = {
                'min': np.min(column_data),
                'max': np.max(column_data),
                'iqr': iqr(column_data),
                'scaler': scaler_type
            }

        return self

    def transform(self, X):
        X_scaled = X.copy()
        for col, scaler in self.scalers.items():
            X_scaled[col] = scaler.transform(X_scaled[col].values.reshape(-1, 1)).flatten()
            
        return X_scaled

    def get_feature_stats(self):
        """For MLOps monitoring and model interpretation"""
        return pd.DataFrame(self.feature_stats).T