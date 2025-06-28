"""
PurgedKFold cross-validator for time series data.

Implements K-Fold splitting with purging and optional embargo windows to prevent
data leakage in time-dependent datasets. Training samples within a specified time
window around the validation fold are excluded (purged) from training.

Classes:
----------
PurgedKFold
    Custom cross-validator with time-based purging and embargo for safer model validation
    on time series data indexed by datetime.

Usage:
----------
- Requires input data X with a datetime index.
- Yields train and validation indices with purged training sets to avoid temporal leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import timedelta

class PurgedKFold:
    def __init__(self, n_splits=5, purging_window=1, embargo_window=0):
        """
        Purged KFold cross-validator that purges training samples within a time window
        around the validation fold.

        Args:
            n_splits (int): Number of folds.
            purging_window (timedelta): Time window before and after validation fold to purge from train.
            embargo_window (timedelta): Additional embargo time after validation fold (optional).
        """
        self.n_splits = n_splits
        self.purging_window = timedelta(days=purging_window)
        self.embargo_window = timedelta(embargo_window)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits"""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and validation sets,
        purging samples in training set that are too close in time to validation set.

        Args:
            X (pd.DataFrame or similar): Input data with datetime index.
            y: Ignored.
            groups: Ignored.

        Yields:
            train_indices, val_indices: np.ndarray of train and validation indices.
        """
        if not hasattr(X, 'index'):
            raise ValueError("Input data X must have a datetime index")

        dates = pd.to_datetime(X.index)
        n_samples = len(X)
        indices = np.arange(n_samples)

        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for train_index, val_index in kf.split(indices):
            val_start_date = dates[val_index[0]]
            val_end_date = dates[val_index[-1]]

            # Calculate purge start/end dates with purge window and embargo
            purge_start_date = val_start_date - self.purging_window
            purge_end_date = val_end_date + self.purging_window + self.embargo_window

            # Create mask for training samples that are outside the purge+embargo window
            train_dates = dates[train_index]
            train_mask = (train_dates < purge_start_date) | (train_dates > purge_end_date)

            purged_train_index = train_index[train_mask]

            yield purged_train_index, val_index
