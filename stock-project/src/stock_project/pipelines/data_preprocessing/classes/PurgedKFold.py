from typing import Tuple, Generator
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import BaseCrossValidator, KFold

class PurgedKFold(BaseCrossValidator):
    def __init__(
        self
        , n_splits=5
        , purging_window=timedelta(days=1)
        , embargo_window=timedelta(0)
    ) -> None:
        """
        Purged K-Fold cross-validator that removes training samples within a time window
        before and after the validation fold.

        Args:
            n_splits (int): Number of folds.
            purging_window (timedelta): Time window before and after validation fold to purge from train.
            embargo_window (timedelta): Additional embargo time after the validation fold.
        """
        self.n_splits = n_splits
        self.purging_window = purging_window
        self.embargo_window = embargo_window

    def get_n_splits(
        self
        , X=None
        , y=None
        , groups=None
    ) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Args:
            X: Ignored.
            y: Ignored.
            groups: Ignored.

        Returns:
            int: Number of folds.
        """
        return self.n_splits

    def split(
        self
        , X
        , y=None
        , groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets,
        purging training samples that are too close in time to the validation set.

        Args:
            X (pd.DataFrame or array-like with DatetimeIndex): Input data with datetime index.
            y: Ignored.
            groups: Ignored.

        Yields:
            (train_indices, val_indices): Tuple of np.ndarrays of train and validation indices.
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

            # Compute purge and embargo window boundaries
            purge_start = val_start_date - self.purging_window
            purge_end = val_end_date + self.purging_window + self.embargo_window

            train_dates = dates[train_index]
            valid_train_mask = (train_dates < purge_start) | (train_dates > purge_end)

            purged_train_index = train_index[valid_train_mask]

            yield purged_train_index, val_index
