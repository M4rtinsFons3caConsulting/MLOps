from prefect import flow, task, get_run_logger
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from stock_project.run_kedro_pipeline import run_pipeline

@task
def run_kedro_pipeline(pipeline_name: str):
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")
    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}")
        raise

@flow(name="Data Unit Test Flow")
def flow_data_unit_tests():
    run_kedro_pipeline("unit_tests")

@flow(name="Full Training Flow")
def flow_production_training():
    run_kedro_pipeline("production_full_train")

@flow(name="Preprocessing Flow")
def flow_preprocess():
    run_kedro_pipeline("preprocess")

@flow(name="Data Splitting Flow")
def flow_splitting():
    run_kedro_pipeline("split")

@flow(name="Training Flow")
def flow_training():
    run_kedro_pipeline("train")

@flow(name="Assessment Flow")
def flow_model_assessment():
    run_kedro_pipeline("assessment")

@flow(name="Full Assessment Flow")
def flow_production_assessment():
    flow_preprocess()
    flow_splitting()
    flow_training()
    flow_model_assessment()

@flow(name="Full Prediction Flow")
def flow_production_prediction():
    run_kedro_pipeline("production_full_prediction")

@flow(name="Drift Flow")
def flow_drift_analysis():
    run_kedro_pipeline("drift")