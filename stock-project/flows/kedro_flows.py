"""
This module defines Prefect flows for running different Kedro pipelines. Each flow corresponds to
a specific stage of the ML lifecycle, such as training, assessment, prediction, and drift analysis.
It integrates Kedro execution within Prefect flows, enabling orchestration, scheduling, and monitoring.
"""

import sys
import os
import traceback

# Ensure UTF-8 output for consistent logging
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from prefect import flow, task, get_run_logger
from src.stock_project.run_kedro_pipeline import run_pipeline

@task
def run_kedro_task(pipeline_name: str):
    """
    Prefect task that runs a specified Kedro pipeline.

    Args:
        pipeline_name (str): The name of the Kedro pipeline to run.

    Raises:
        Exception: Re-raises any exceptions encountered during pipeline execution.
    """
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")
    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}\n{traceback.format_exc()}")
        raise

@flow(name="Data Unit Test Flow")
def flow_data_unit_tests():
    """
    Prefect flow for running unit tests on data pipelines.
    Useful for validating data transformations and ensuring integrity.
    """
    run_kedro_task("unit_tests")

@flow(name="Full Training Flow")
def flow_production_training():
    """
    Prefect flow for running the full production training pipeline.
    Includes data preparation, model training, and related tasks.
    """
    run_kedro_task("production_full_train")

@flow(name="Full Assessment Flow")
def flow_production_assessment():
    """
    Prefect flow for running a full model assessment workflow.
    Runs preprocessing, data splitting, training, and final assessment pipelines sequentially.
    """
    run_kedro_task("preprocess")
    run_kedro_task("split")
    run_kedro_task("train")
    run_kedro_task("assessment")

@flow(name="Full Prediction Flow")
def flow_production_prediction():
    """
    Prefect flow for running the full prediction pipeline in production.
    Generates predictions using the latest trained models.
    """
    run_kedro_task("production_full_prediction")

@flow(name="Drift Flow")
def flow_drift_analysis():
    """
    Prefect flow for running data or model drift analysis.
    Helps monitor and identify deviations in data or model performance over time.
    """
    run_kedro_task("drift")
