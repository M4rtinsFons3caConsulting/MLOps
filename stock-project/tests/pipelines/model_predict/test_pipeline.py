"""
Unit tests for the `model_predict` node.

Tests:
1. Logs the prediction creation info message
2. Returns a pd.Series indexed by the input X_test
"""

import pytest
import pandas as pd
from unittest import mock
import logging
import model_predict  # The module containing make_predictions

@pytest.fixture
def sample_X_test():
    return pd.DataFrame({
        "feature1": [5, 6, 7],
        "feature2": [50, 60, 70]
    }, index=pd.date_range("2023-01-01", periods=3))

@pytest.fixture
def dummy_pipeline():
    pipeline = mock.MagicMock()
    pipeline.predict.return_value = [1, 0, 1]
    return pipeline

### ---------- Logger logs ---------- ###

def test_logs_prediction_created(sample_X_test, dummy_pipeline, caplog):

    with caplog.at_level(logging.INFO):
        model_predict.make_predictions(sample_X_test, dummy_pipeline)
    assert "Predictions created." in caplog.text

### ---------- Returns predict series ---------- ###

def test_returns_series_with_correct_index(sample_X_test, dummy_pipeline):

    result = model_predict.make_predictions(sample_X_test, dummy_pipeline)
    assert isinstance(result, pd.Series)
    assert all(result.index == sample_X_test.index)
    assert result.name == "y_pred"
