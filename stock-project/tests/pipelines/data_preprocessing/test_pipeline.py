"""
Unit tests for the data_preprocessing pipeline (core coverage):

Test for `apply_indicators_to_group`:
1. Returns DataFrame with additional indicator columns

Test for `perform_feature_engineering`:
2. Applies `apply_indicators_to_group` per symbol

Test for `create_target`:
3. Creates binary `label` column correctly

Test for `widden_df`:
4. Column names follow `{symbol}_{feature}` format

Test for `handle_missing_values`:
5. Drops all remaining rows with NaNs

Test for `prepare_model_input`:
6. Merges features and labels on `date`
"""

import pandas as pd
import numpy as np
from unittest import mock

from src.stock_project.pipelines.data_preprocessing.nodes import (
    apply_indicators_to_group,
    perform_feature_engineering,
    create_target,
    widden_df,
    handle_missing_values,
    prepare_model_input
)


### -----------  Test for `apply_indicators_to_group` ----------- ###

def test_returns_df_with_additional_indicators(test_ohlcv_df):
    result = apply_indicators_to_group(test_ohlcv_df)

    assert "EMA_10" in result.columns
    assert "RSI_14" in result.columns
    assert "MACD" in result.columns
    assert len(result) == len(test_ohlcv_df)


### -----------  Test for `perform_feature_engineering` ----------- ###

def test_applies_indicators_per_symbol(test_grouped_df):
    with mock.patch(
        "src.stock_project.pipelines.data_preprocessing.nodes.apply_indicators_to_group"
    ) as mock_apply:
        mock_apply.side_effect = lambda df: df.assign(dummy_feature=1)

        result, _ = perform_feature_engineering(test_grouped_df, versions={}, is_to_feature_store=False)

        assert "dummy_feature" in result.columns
        assert result["dummy_feature"].nunique() == 1


### -----------  Test for `create_target` ----------- ###

def test_creates_binary_label_column(test_feature_df):
    result, _ = create_target(
        test_feature_df,
        versions={},
        prediction_horizon=1,
        is_to_feature_store=False
    )

    assert "label" in result.columns
    assert set(result["label"].dropna().unique()).issubset({0, 1})


### -----------  Test for `widden_df` ----------- ###

def test_column_names_follow_symbol_feature_format(test_long_df):
    result = widden_df(test_long_df)

    for col in result.columns:
        if col != "date":
            assert "_" in col
            parts = col.split("_")
            assert len(parts) >= 2


### -----------  Test for `handle_missing_values` ----------- ###

def test_drops_all_remaining_rows_with_nans(test_df_with_nans):
    result = handle_missing_values(test_df_with_nans)

    assert result.isna().sum().sum() == 0
    assert not any("SUPERT" in col and "SUPERTd" not in col for col in result.columns)


### -----------  Test for `prepare_model_input` ----------- ###

def test_merges_features_and_labels_on_date(test_grouped_df, test_feature_df):
    with mock.patch(
        "src.stock_project.pipelines.data_preprocessing.nodes.perform_feature_engineering"
    ) as mock_features, \
    mock.patch(
        "src.stock_project.pipelines.data_preprocessing.nodes.create_target"
    ) as mock_target, \
    mock.patch(
        "src.stock_project.pipelines.data_preprocessing.nodes.widden_df"
    ) as mock_wide, \
    mock.patch(
        "src.stock_project.pipelines.data_preprocessing.nodes.handle_missing_values"
    ) as mock_clean:

        features_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "AAPL_feature1": [1, 2, 3]
        })

        target_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "label": [1, 0, 1]
        })

        mock_features.return_value = (features_df, {})
        mock_target.return_value = (target_df, {})
        mock_wide.return_value = features_df
        mock_clean.return_value = features_df

        result, _ = prepare_model_input(
            test_grouped_df,
            versions={},
            is_to_feature_store=False
        )

        assert "label" in result.columns
        assert "AAPL_feature1" in result.columns
        assert len(result) <= 3
