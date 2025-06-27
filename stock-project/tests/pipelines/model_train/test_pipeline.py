"""
Unit tests for the `model_train` node in the training pipeline.

Tests for `model_train`:

1. Trains pipeline using `pipeline.fit(X_train, y_train)`  
2. Logs model to MLflow using `mlflow.sklearn.log_model()`  
"""

import pytest
import pandas as pd
import numpy as np
from unittest import mock
import matplotlib.pyplot as plt
import model_train  # The module containing the `model_train` function

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40]
    })
    y = pd.DataFrame({"target": [0, 1, 0, 1]})
    return X, y

@pytest.fixture
def dummy_pipeline():
    pipeline = mock.MagicMock()
    pipeline.named_steps = {"model": mock.MagicMock(), "feature_selection": mock.Mock()}
    pipeline.steps = [("model", pipeline.named_steps["model"])]
    pipeline.named_steps["feature_selection"].get_feature_names_out.return_value = ["feature1", "feature2"]
    return pipeline

### ----------- Test for `pipeline_fit` ----------- ###


# 1. Trains pipeline using pipeline.fit(X_train, y_train)
def test_trains_pipeline_fit_called(sample_data, dummy_pipeline):
    X_train, y_train = sample_data
    model_train.model_train(X_train, y_train, dummy_pipeline)
    dummy_pipeline.fit.assert_called_once_with(X_train, np.ravel(y_train))


### ----------- Test for `sklearn.log_model` ----------- ###


# 2. Logs model to MLflow using mlflow.sklearn.log_model()
def test_logs_model_to_mlflow(sample_data, dummy_pipeline):
    X_train, y_train = sample_data
    with mock.patch("model_train.mlflow.sklearn.log_model") as mock_log_model, \
         mock.patch("model_train.mlflow.get_experiment_by_name") as mock_get_exp, \
         mock.patch("model_train.mlflow.start_run"), \
         mock.patch("model_train.yaml.load") as mock_yaml_load, \
         mock.patch("builtins.open", mock.mock_open(read_data="tracking:\n  experiment:\n    name: test-exp")):

        mock_yaml_load.return_value = {'tracking': {'experiment': {'name': 'test-exp'}}}
        mock_get_exp.return_value = mock.Mock(experiment_id="123")
        model_train.model_train(X_train, y_train, dummy_pipeline)
        mock_log_model.assert_called_once_with(dummy_pipeline, artifact_path="trained_pipeline")
