import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.stock_project.pipelines.data_preprocessing.nodes import prepare_model_input

@pytest.fixture
def raw_data():
    """Raw input data for feature engineering."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-02",
            "2024-01-02", "2024-01-03", "2024-01-03"
        ]),
        "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
        "open": [100, 200, 101, 201, 102, 202],
        "high": [110, 210, 111, 211, 112, 212],
        "low": [90, 190, 91, 191, 92, 192],
        "close": [105, 205, 106, 206, 107, 207],
        "volume": [1000, 2000, 1100, 2100, 1200, 2200],
        "transformation": [0.1, np.nan, 0.15, 0.2, 0.25, 0.3]
    })

@pytest.fixture
def engineered_data(raw_data):
    """Simulate engineered data output with NaNs dropped."""
    df = raw_data.copy()
    df_wide = df.pivot(index='date', columns='symbol')
    df_wide.columns = [f"{symbol}_{feature}" for feature, symbol in df_wide.columns]
    df_wide = df_wide.reset_index()
    df_wide = df_wide.dropna().reset_index(drop=True)  # Drop NaNs as per handle_missing_values
    return df_wide

@pytest.fixture
def data_labels():
    """Simulated labels aligned with engineered features."""
    return pd.DataFrame({
        'date': pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        'label': [1, 0, 1]
    })

def test_prepare_model_input(raw_data, engineered_data, data_labels):
    
    with patch('src.stock_project.pipelines.data_preprocessing.nodes.perform_feature_engineering') as mock_feat_eng, \
         patch('src.stock_project.pipelines.data_preprocessing.nodes.create_target') as mock_create_target, \
         patch('src.stock_project.pipelines.data_preprocessing.nodes.widden_df') as mock_widden_df, \
         patch('src.stock_project.pipelines.data_preprocessing.nodes.handle_missing_values') as mock_handle_missing:

        mock_feat_eng.return_value = (engineered_data, {'feature_engineering': 1})
        mock_create_target.return_value = (data_labels, {'target_creation': 2})
        mock_widden_df.return_value = engineered_data
        mock_handle_missing.return_value = engineered_data

        model_data, versions = prepare_model_input(raw_data)

        assert 'date' in model_data.columns
        assert 'label' in model_data.columns
        assert not model_data['label'].isna().any()

        expected_versions = {
            'feature_engineering': 1,
            'target_creation': 2
        }
        assert versions == expected_versions
