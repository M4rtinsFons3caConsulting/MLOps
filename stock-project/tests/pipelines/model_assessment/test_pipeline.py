"""
Unit tests for the optuna_model_search framework (core coverage):

Test for `build_pipeline`:
1. Constructs pipeline with correct steps and model instance

Test for `run_optuna_search`:
2. Returns estimator, params, and score with proper types

Test for `find_champion_model`:
3. Returns best estimator after multi-model hyperparameter search

Test for `FeatureScaler`:
4. Detects and applies appropriate scaler per feature

Test for `KBestRFESelector`:
5. Selects features combining SelectKBest and RFE correctly

Test for `PurgedKFold`:
6. Generates train/validation splits with proper purging and embargo
"""

import pytest
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from optuna.distributions import IntDistribution, FloatDistribution

from optuna_model_search import (
    build_pipeline,
    run_optuna_search,
    find_champion_model,
    FeatureScaler,
    KBestRFESelector,
    PurgedKFold,
)

# Sample data generation for tests
@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.linspace(0, 100, 100),
        "feature3": np.random.normal(0, 1, 100),
    }, index=pd.date_range(start="2021-01-01", periods=100))

    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


### ----------- Test for `build_pipeline` ----------- ###

def test_build_pipeline_structure_and_steps():
    pipeline = build_pipeline("LogisticRegression", {"max_iter": 50})
    steps = dict(pipeline.named_steps)
    assert "feature_selection" in steps
    assert "scaler" in steps
    assert "model" in steps
    assert isinstance(steps["model"], LogisticRegression)
    assert steps["model"].max_iter == 50

def test_build_pipeline_model_instance_type():
    pipeline = build_pipeline("RandomForestClassifier", {"n_estimators": 10})
    model = pipeline.named_steps["model"]
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10


### ----------- Test for `run_optuna_search` ----------- ###

def test_run_optuna_search_returns_expected_tuple(sample_data):
    X, y = sample_data
    model_init_params = {"max_iter": 10}
    hyperparam_space = {
        "model__C": {"distribution": "float", "low": 0.01, "high": 1.0, "log": True}
    }
    cv_args = {"n_splits": 3, "purging_window": 1, "embargo_window": 0}
    estimator, params, score = run_optuna_search(
        "LogisticRegression",
        X,
        y,
        model_init_params,
        hyperparam_space,
        n_trials=5,
        scoring_metric="accuracy",
        cv_args=cv_args,
        random_state=42,
        verbose=0,
        n_jobs=1,
    )
    assert hasattr(estimator, "predict")
    assert isinstance(params, dict)
    assert isinstance(score, float)

def test_run_optuna_search_hyperparam_conversion(sample_data):
    X, y = sample_data
    hyperparam_space = {
        "model__n_estimators": {"distribution": "int", "low": 5, "high": 15},
        "model__max_depth": {"distribution": "int", "low": 2, "high": 5},
    }
    cv_args = {"n_splits": 2, "purging_window": 1, "embargo_window": 0}
    # run without errors, implying conversion to distributions succeeded
    estimator, params, score = run_optuna_search(
        "RandomForestClassifier",
        X,
        y,
        {"n_jobs": 1, "random_state": 0},
        hyperparam_space,
        n_trials=3,
        scoring_metric="f1_score",
        cv_args=cv_args,
        random_state=0,
        verbose=0,
        n_jobs=1,
    )
    assert all(isinstance(value, (int, float)) for value in params.values())


### ----------- Test for `find_champion_model` ----------- ###

def test_find_champion_model_returns_best_estimator(sample_data):
    X, y = sample_data
    model_init_params_dict = {
        "LogisticRegression": {"max_iter": 20},
        "RandomForestClassifier": {"n_estimators": 10, "random_state": 1},
    }
    hyperparam_spaces_dict = {
        "LogisticRegression": {
            "model__C": {"distribution": "float", "low": 0.01, "high": 1.0, "log": True}
        },
        "RandomForestClassifier": {
            "model__n_estimators": {"distribution": "int", "low": 5, "high": 15}
        },
    }
    cv_args = {"n_splits": 3, "purging_window": 1, "embargo_window": 0}

    best_estimator = find_champion_model(
        X,
        y,
        model_init_params_dict,
        hyperparam_spaces_dict,
        n_trials=3,
        scoring_metric="accuracy",
        cv_args=cv_args,
        kwargs={"random_state": 42, "verbose": 0, "n_jobs": 1},
    )
    assert hasattr(best_estimator, "predict")

def test_find_champion_model_mlflow_logging(monkeypatch, sample_data):
    X, y = sample_data

    # Mock mlflow methods to test logging without side effects
    class DummyRun:
        info = type("info", (), {"run_id": "test_run_id"})
    class DummyMLflow:
        def get_experiment_by_name(self, name):
            return type("exp", (), {"experiment_id": "dummy_id"})
        def start_run(self, experiment_id=None, nested=False):
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummyContext()
        def log_params(self, params): pass
        def log_metric(self, key, value): pass
        def last_active_run(self):
            return DummyRun()

    monkeypatch.setattr("optuna_model_search.mlflow", DummyMLflow())

    model_init_params_dict = {
        "LogisticRegression": {"max_iter": 5},
    }
    hyperparam_spaces_dict = {
        "LogisticRegression": {
            "model__C": {"distribution": "float", "low": 0.1, "high": 1.0, "log": True}
        },
    }
    cv_args = {"n_splits": 2, "purging_window": 1, "embargo_window": 0}

    best_estimator = find_champion_model(
        X,
        y,
        model_init_params_dict,
        hyperparam_spaces_dict,
        n_trials=2,
        scoring_metric="accuracy",
        cv_args=cv_args,
        kwargs={"random_state": 0, "verbose": 0, "n_jobs": 1},
    )
    assert hasattr(best_estimator, "predict")


### ----------- Test for `FeatureScaler` ----------- ###

def test_feature_scaler_fit_assigns_correct_scalers():
    X = pd.DataFrame({
        "bounded_feature": np.linspace(0, 100, 50),
        "outlier_feature": np.concatenate([np.random.normal(0, 1, 45), np.array([15, 20, 25, 30, 35])]),
        "normal_feature": np.random.normal(0, 1, 50),
    })
    scaler = FeatureScaler().fit(X)
    stats = scaler.get_feature_stats()
    assert "bounded_feature" in scaler.scalers
    assert stats.loc["bounded_feature", "scaler"] == "minmax"
    assert stats.loc["outlier_feature", "scaler"] == "robust"
    assert stats.loc["normal_feature", "scaler"] == "standard"

def test_feature_scaler_auto_detect_scaler_behavior():
    scaler = FeatureScaler()
    bounded_data = np.linspace(0, 100, 50).reshape(-1, 1)
    robust_data = np.concatenate([np.random.normal(0, 1, 45), np.array([15, 20, 25, 30, 35])]).reshape(-1, 1)
    standard_data = np.random.normal(0, 1, 50).reshape(-1, 1)

    s1, t1 = scaler._auto_detect_scaler(bounded_data)
    s2, t2 = scaler._auto_detect_scaler(robust_data)
    s3, t3 = scaler._auto_detect_scaler(standard_data)

    assert t1 == "minmax"
    assert t2 == "robust"
    assert t3 == "standard"


### ----------- Test for `KBestRFESelector` ----------- ###

def test_kbestrfe_selector_fit_and_feature_selection(sample_data):
    X, y = sample_data
    selector = KBestRFESelector(k=3, n_features_to_select=2, random_state=0)
    selector.fit(X, y)
    selected_features = selector.get_feature_names_out()
    assert len(selected_features) == 2
    assert all(f in X.columns for f in selected_features)

def test_kbestrfe_selector_transform_output(sample_data):
    X, y = sample_data
    selector = KBestRFESelector(k=4, n_features_to_select=2, random_state=0)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    assert isinstance(X_transformed, pd.DataFrame)
    assert list(X_transformed.columns) == list(selector.get_feature_names_out())
    assert X_transformed.shape[1] == 2
    assert X_transformed.shape[0] == X.shape[0]


### ----------- Test for `PurgedKFold` ----------- ###

def test_purgedkfold_split_and_purge_behavior():
    dates = pd.date_range(start="2021-01-01", periods=20)
    X = pd.DataFrame(np.random.rand(20, 2), index=dates)
    pkf = PurgedKFold(n_splits=4, purging_window=1, embargo_window=1)

    splits = list(pkf.split(X))
    assert len(splits) == 4

    for train_idx, val_idx in splits:
        train_dates = X.index[train_idx]
        val_dates = X.index[val_idx]
        purge_start = val_dates[0] - timedelta(days=1)
        purge_end = val_dates[-1] + timedelta(days=2)  # purging_window + embargo_window
        # All training dates should lie outside the purge window
        assert all((d < purge_start or d > purge_end) for d in train_dates)

def test_purgedkfold_get_n_splits():

    pkf = PurgedKFold(n_splits=7)
    
    assert pkf.get_n_splits() == 7
