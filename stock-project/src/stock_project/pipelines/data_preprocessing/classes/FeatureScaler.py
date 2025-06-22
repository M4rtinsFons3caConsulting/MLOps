import numpy as np
import pandas as pd

from scipy.stats import iqr, kurtosis, skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, overrides=None):
        self.scalers = {}
        self.feature_stats = {}
        self.overrides = overrides or {}

    def _parse_overrides(self):
        scaler_map = {
            'minmax': MinMaxScaler(feature_range=(0, 1)),
            'robust': RobustScaler(),
            'standard': StandardScaler()
        }
        parsed = {}
        for feature, val in self.overrides.items():
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in scaler_map:
                    parsed[feature] = scaler_map[val_lower]
                else:
                    raise ValueError(f"Unknown scaler override '{val}' for feature '{feature}'")
            else:
                parsed[feature] = val
        return parsed

    def _auto_detect_scaler(self, data):
        data = data.flatten()
        q1, q3 = np.percentile(data, [25, 75])
        iqr_val = q3 - q1
        range_val = np.max(data) - np.min(data)
        has_outliers = (iqr_val > 0) and (range_val / iqr_val > 4)
        
        skewness = skew(data)
        kurt = kurtosis(data)

        is_bounded = (np.min(data) >= 0) and (np.max(data) <= 100) and (range_val >= 50)

        if is_bounded:
            return MinMaxScaler(feature_range=(0, 1)), 'minmax'
        elif abs(skewness) > 1 or kurt > 3 or has_outliers:
            return RobustScaler(), 'robust'
        else:
            return StandardScaler(), 'standard'

    def fit(self, X, y=None):
        parsed_overrides = self._parse_overrides()

        for col in X.columns:
            column_data = X[col].values.reshape(-1, 1)

            if col in parsed_overrides:
                scaler = parsed_overrides[col]
                scaler_type = type(scaler).__name__.replace('Scaler', '').lower()
            else:
                scaler, scaler_type = self._auto_detect_scaler(column_data)

            self.scalers[col] = scaler.fit(column_data)
            self.feature_stats[col] = {
                'min': np.min(column_data),
                'max': np.max(column_data),
                'iqr': iqr(column_data.flatten()),
                'skewness': skew(column_data.flatten()),
                'kurtosis': kurtosis(column_data.flatten()),
                'scaler': scaler_type
            }
        return self

    def transform(self, X):
        X_scaled = X.copy()
        for col, scaler in self.scalers.items():
            X_scaled[col] = scaler.transform(X_scaled[col].values.reshape(-1, 1)).flatten()
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_stats(self):
        return pd.DataFrame(self.feature_stats).T
