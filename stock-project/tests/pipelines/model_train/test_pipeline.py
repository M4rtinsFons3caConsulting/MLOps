import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch, mock_open, MagicMock

from src.stock_project.pipelines.model_train.nodes import model_train

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=10)
    X_train = pd.DataFrame({
        "feature1": range(10),
        "feature2": range(10, 20)
    }, index=dates)
    y_train = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)), index=dates)
    return X_train, y_train

@pytest.fixture
def pipeline():
    return Pipeline([
        ("feature_selection", SelectKBest(k=2)),
        ("model", LogisticRegression())
    ])

@patch("builtins.open", new_callable=mock_open, read_data="""
tracking:
  experiment:
    name: stock_project
""")
@patch("yaml.load", return_value={"tracking": {"experiment": {"name": "stock_project"}}})
@patch("mlflow.get_experiment_by_name")
@patch("mlflow.start_run")
@patch("mlflow.sklearn.log_model")
@patch("shap.Explainer")
@patch("shap.summary_plot")
def test_model_train_returns_pipeline_and_figure(
    mock_shap_summary_plot,
    mock_shap_explainer,
    mock_log_model,
    mock_start_run,
    mock_get_experiment,
    mock_yaml_load,
    mock_open_file,
    sample_data,
    pipeline,
):
    mock_get_experiment.return_value = MagicMock(experiment_id="123")
    mock_start_run.return_value.__enter__.return_value = None

    mock_shap = MagicMock()
    mock_shap_explainer.return_value = mock_shap
    mock_shap_summary_plot.return_value = None 

    trained_pipeline, _ = model_train(*sample_data, pipeline)

    assert hasattr(trained_pipeline.named_steps["model"], "coef_")
 