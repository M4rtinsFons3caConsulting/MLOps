"""
This is a pipeline 'model_train'
generated using Kedro 0.19.14
"""

from typing import Tuple

import pandas as pd
import numpy as np
import pickle
import mlflow
import yaml
import shap
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    pipeline: pickle,
) -> Tuple[pickle, plt.Figure]:
    """
    Trains a scikit-learn pipeline, logs metrics to MLflow, and generates a SHAP summary plot.

    This function performs the following steps:
    - Loads the MLflow experiment configuration.
    - Sets the index of input DataFrames to 'date' for alignment.
    - Trains the provided pipeline on the training data.
    - Evaluates the trained pipeline on both training and test sets.
    - Logs accuracy metrics to MLflow.
    - Logs the trained model to MLflow.
    - Computes SHAP values (if supported) and generates a SHAP summary plot.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.DataFrame): Training target vector.
        y_test (pd.DataFrame): Testing target vector.
        pipeline (pickle): A fitted scikit-learn pipeline that includes preprocessing and a final estimator.

    Returns:
        Tuple[pickle, plt.Figure]: The trained pipeline and a matplotlib figure object with the SHAP summary plot.
    """
    # Load experiment name from config
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Set indices
    X_train.set_index('date', inplace=True)
    X_test.set_index('date', inplace=True)
    y_train.set_index('date', inplace=True)
    y_test.set_index('date', inplace=True)

    # Flatten target for sklearn compatibility
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        acc_train = np.mean(y_train_pred == y_train)
        acc_test = np.mean(y_test_pred == y_test)

        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="trained_pipeline")

        logger.info(f"✅ Model trained and logged. Test accuracy: {acc_test:.4f}")

        # SHAP output
        shap_values = None
        try:
            model = pipeline.named_steps.get("model") or pipeline.steps[-1][1]
            feature_names = pipeline.named_steps['feature_selection'].get_feature_names_out()

            explainer = shap.Explainer(model)
            shap_values = explainer(X_train[feature_names])


            shap.initjs()

            plt.figure(figsize=(10, 6))
            # SHAP summary plot (matplotlib figure)
            shap.summary_plot(shap_values, X_train[feature_names], feature_names=feature_names, show=False)
            fig = plt.gcf()
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"SHAP skipped: {e}")

    return pipeline, plt
