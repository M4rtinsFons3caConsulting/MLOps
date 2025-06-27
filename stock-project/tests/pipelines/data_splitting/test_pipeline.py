"""
Unit tests for the data_splitting pipeline (core coverage):

Test for `split_data`:
1. Returns four outputs: X_train, X_test, y_train, y_test
2. Data is sorted by 'date' before splitting
3. MLflow logs correct parameters and metrics for split size
"""

import pandas as pd
import pytest
from unittest import mock

from src.stock_project.pipelines.data_splitting.nodes import split_data


### ----------- Fixtures ----------- ###

@pytest.fixture
def test_df_split_ready():
    return pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=10),
        "feature1": range(10),
        "label": [0, 1] * 5
    })


@pytest.fixture
def test_df_unsorted():
    df = pd.DataFrame({
        "date": pd.date_range(start="2022-01-01", periods=10).tolist()[::-1],
        "feature1": range(10),
        "label": [1, 0] * 5
    })
    return df


### ----------- Test 1: Returns four outputs ----------- ###

def test_returns_four_outputs(test_df_split_ready):
    X_train, X_test, y_train, y_test = split_data(test_df_split_ready)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


### ----------- Test 2: Data is sorted by 'date' ----------- ###

def test_data_is_sorted_by_date_before_splitting(test_df_unsorted):
    # Copy to preserve input
    df_copy = test_df_unsorted.copy()

    split_data(df_copy)

    # Check if the DataFrame was sorted in-place and index set to 'date'
    sorted_dates = df_copy['date'].sort_values().values
    assert all(df_copy.index.values == sorted_dates)


### ----------- Test 3: MLflow logs correct parameters and metrics ----------- ###

def test_mlflow_logging(test_df_split_ready):
    with mock.patch("src.stock_project.pipelines.data_splitting.nodes.mlflow") as mock_mlflow:
        split_data(test_df_split_ready, test_size=0.3)

        mock_mlflow.log_param.assert_called_with("test_size", 0.3)
        assert mock_mlflow.log_metric.call_count == 2

        call_args = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert "train_size" in call_args
        assert "test_size" in call_args
