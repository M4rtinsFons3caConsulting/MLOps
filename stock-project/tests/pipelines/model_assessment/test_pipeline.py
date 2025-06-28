import pytest
from unittest.mock import MagicMock, patch, mock_open
from sklearn.pipeline import Pipeline
from src.stock_project.pipelines.model_assessment.nodes import find_champion_model 

def test_find_champion_model_returns_fitted_pipeline():
    # Sample inputs
    X = MagicMock()
    y = MagicMock()
    model_init_params_dict = {"model_a": {"param": 1}}
    hyperparam_spaces_dict = {"model_a": {"param": {"distribution": "uniform", "low": 0, "high": 1}}}
    n_trials = 1
    scoring_metric = "accuracy"
    cv_args = {"n_splits": 2}
    kwargs = {"random_state": 42}

    # Create a mock fitted Pipeline object with a fitted attribute
    mock_pipeline = Pipeline(steps=[])
    mock_pipeline.fitted_ = True  # dummy attribute to mark "fitted"

    # Mock for mlflow.last_active_run().info.run_id
    mock_run_info = MagicMock()
    mock_run_info.run_id = "mock_run_id"
    mock_last_active_run = MagicMock()
    mock_last_active_run.info = mock_run_info

    m = mock_open(read_data="tracking:\n  experiment:\n    name: test\n")

    with patch("builtins.open", m), \
         patch("src.stock_project.pipelines.model_assessment.nodes.run_optuna_search", return_value=(mock_pipeline, {}, 0.9)), \
         patch("src.stock_project.pipelines.model_assessment.nodes.mlflow.get_experiment_by_name"), \
         patch("src.stock_project.pipelines.model_assessment.nodes.mlflow.start_run"), \
         patch("src.stock_project.pipelines.model_assessment.nodes.mlflow.log_params"), \
         patch("src.stock_project.pipelines.model_assessment.nodes.mlflow.log_metric"), \
         patch("src.stock_project.pipelines.model_assessment.nodes.mlflow.last_active_run", return_value=mock_last_active_run), \
         patch("src.stock_project.pipelines.model_assessment.nodes.yaml.load", return_value={"tracking": {"experiment": {"name": "test"}}}):

        best_estimator = find_champion_model(
            X, y,
            model_init_params_dict=model_init_params_dict,
            hyperparam_spaces_dict=hyperparam_spaces_dict,
            n_trials=n_trials,
            scoring_metric=scoring_metric,
            cv_args=cv_args,
            kwargs=kwargs
        )

    # Assert the output is a Pipeline instance
    assert isinstance(best_estimator, Pipeline)

    # Assert it has the dummy fitted attribute
    assert hasattr(best_estimator, "fitted_")
