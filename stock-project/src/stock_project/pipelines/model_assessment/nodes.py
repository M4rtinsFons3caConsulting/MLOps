"""
optuna_model_search.py

This module provides a framework for automated hyperparameter optimization of machine learning models
using Optuna's integration with scikit-learn (OptunaSearchCV). 

It currently supports sklearn's Logistic Regression, Random Forest, and XGBoost.

Key features:
- Construction of sklearn pipelines with custom feature selectors and scalers.
- Flexible hyperparameter tuning using Optuna with purged cross-validation (PurgedKFold).
- Support for multiple classification models with customizable initialization parameters and hyperparameter spaces.
- Logging of tuning parameters and results with MLflow for experiment tracking.
- Scoring flexibility with multiple classification metrics (F1, accuracy, recall, precision, ROC AUC, AUPR).

Main functions:
- build_pipeline: Constructs a pipeline for a specified model with feature selection and scaling.
- run_optuna_search: Executes OptunaSearchCV to find the best hyperparameters for a given pipeline and data.
- find_champion_model: Runs optimization across multiple models and returns the best estimator and scores.

Dependencies:
- scikit-learn
- xgboost
- optuna
- mlflow
- custom modules: PurgedKFold, KBestRFESelector, FeatureScaler

"""

from datetime import timedelta
from typing import Any, Dict, Tuple

import yaml
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.feature_selection import f_classif

from .classes.PurgedKFold import PurgedKFold
from .classes.FeatureSelector import KBestRFESelector
from .classes.DynamicScaler import FeatureScaler

import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution

import logging
import mlflow

logger = logging.getLogger(__name__)

SCORER_MAPPING = {
    "f1_score": f1_score,
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "roc_auc": roc_auc_score,
    "aupr": average_precision_score,
}

MODEL_CLASS_MAPPING = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

OPTUNA_DISTRIBUTION_MAPPING = {
    "int": IntDistribution
    ,"float": FloatDistribution
}


def build_pipeline(model_name: str, model_init_params: Dict[str, Any]) -> Pipeline:
    """
    Build sklearn pipeline with feature selection, scaling, and the specified model initialized.

    Args:
        model_name (str): Model class name string.
        model_init_params (dict): Parameters to initialize the model.

    Returns:
        Pipeline: sklearn pipeline object.
    """
    model_cls = MODEL_CLASS_MAPPING[model_name]
    model_instance = model_cls(**model_init_params)

    pipeline = Pipeline(
        [
            ("feature_selection", KBestRFESelector(score_func=f_classif)),
            ("scaler", FeatureScaler()),
            ("model", model_instance),
        ]
    )

    return pipeline


def run_optuna_search(
    model_name: str,
    X,
    y,
    model_init_params: Dict[str, Any],
    hyperparam_space: Dict[str, Any],
    n_trials: int,
    scoring_metric: str,
    cv_args: Dict[str, Any],
    **kwargs,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Run OptunaSearchCV to tune hyperparameters on a pipeline built for given model.

    Args:
        model_name (str): Model class name string.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        model_init_params (dict): Model initialization parameters.
        hyperparam_space (dict): Hyperparameter search space.
        n_trials (int): Number of Optuna trials.
        scoring_metric (str): Scoring metric key from SCORER_MAPPING.
        cv_args (dict): Arguments for PurgedKFold, unpacked.
        **kwargs: Additional keyword arguments (expects "random_state").

    Returns:
        Tuple[Pipeline, dict, float]: best_estimator, best_params, best_score
    """
    pipeline = build_pipeline(model_name, model_init_params)
    scoring = make_scorer(SCORER_MAPPING[scoring_metric], average="macro")

    random_state = kwargs.get("random_state")
    verbose = kwargs.get("verbose")
    n_jobs = kwargs.get("n_jobs")

    # Convert numeric range lists to Optuna distributions
    hyperparam_space = {
        param: OPTUNA_DISTRIBUTION_MAPPING[config["distribution"]](
            low=config["low"],
            high=config["high"],
            log=config.get("log", False)
        )
        for param, config in hyperparam_space.items()
    }

    search = OptunaSearchCV(
        pipeline,
        hyperparam_space,
        cv=PurgedKFold(**cv_args),
        n_trials=n_trials,
        scoring=scoring,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_params_, search.best_score_


def find_champion_model(
    X,
    y,
    model_init_params_dict: Dict[str, Dict[str, Any]],
    hyperparam_spaces_dict: Dict[str, Dict[str, Any]],
    n_trials: int,
    scoring_metric: str,
    cv_args: Dict[str, Any],
    kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Optuna-based optimization for all models passed via parameters dicts.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        model_init_params_dict (dict): Mapping model names to init parameters.
        hyperparam_spaces_dict (dict): Mapping model names to hyperparameter search spaces.
        n_trials (int): Number of trials per model.
        scoring_metric (str): Scoring metric key.
        cv_args (dict): Arguments for PurgedKFold, unpacked.
        kwargs: Additional keyword arguments (expects "random_state").

    Returns:
        dict: Mapping model names to dicts containing 'estimator', 'params', 'score'.
    """
    X.set_index('date', inplace=True)
    y.set_index('date', inplace=True)

    results = {}

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)

    for model_name in model_init_params_dict.keys():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):

            logger.info(f"🚀 Starting optimization for model: {model_name}")

            best_estimator, best_params, best_score = run_optuna_search(
                model_name,
                X,
                np.ravel(y),
                model_init_params=model_init_params_dict[model_name],
                hyperparam_space=hyperparam_spaces_dict[model_name],
                n_trials=n_trials,
                scoring_metric=scoring_metric,
                cv_args=cv_args,
                **kwargs,
            )

            logger.info(f"🎯 Completed optimization for {model_name} | Best score: {best_score:.4f}")
            mlflow.log_params({f"{model_name}_init_params": model_init_params_dict[model_name]})
            mlflow.log_params({f"{model_name}_hyperparams": best_params})
            mlflow.log_metric(f"{model_name}_best_score", best_score)

            results[model_name] = {
                "estimator": best_estimator,
                "params": best_params,
                "score": best_score,
            }

            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")

    logger.info("✅ All model optimizations completed.")

    # Get the best model by score
    best_model_name = max(results, key=lambda name: results[name]["score"])
    best_estimator = results[best_model_name]["estimator"]

    logger.info(f"🏆 Best model: {best_model_name} with score {results[best_model_name]['score']:.4f}")

    return best_estimator
