"""
Pipeline 'data_splitting' node for data splitting into training, and hold-out.

This module provides functionality to split time series data into training and testing sets
while preserving chronological order. It logs split parameters and dataset sizes using MLflow.

Functions:
- split_data: Splits data into train/test sets without shuffling, suitable for time series.

Inputs:
- data: DataFrame containing features, a 'label' column, and a 'date' column.
- test_size: Fraction of data reserved for testing (default 0.2).

Outputs:
- X_train, X_test: Feature subsets for training and testing.
- y_train, y_test: Label subsets for training and testing.
"""

import logging

import pandas as pd

from sklearn.model_selection import train_test_split

import mlflow

# Logger
logger = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame
    ,test_size=0.2
):
    """
    Splits time series data into training and testing sets while preserving chronological order.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the features and a 'label' column for the target. 
        Must contain a 'date' column for sorting.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Testing labels.
    """
    # Sort and set index
    data.sort_values(by='date', inplace=True)
    data.set_index('date', inplace=True)

    logger.info(f"Starting data split with test_size={test_size}")

    # Perform the split
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['label'])
        ,data['label']
        ,test_size=test_size
        ,shuffle=False  # working with time series
    )

    logger.info(f"Split complete: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

    # Track parameters and metrics with MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))

    return X_train, X_test, y_train, y_test