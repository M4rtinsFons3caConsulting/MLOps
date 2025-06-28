"""Project pipelines.

This script registers and organizes all Kedro pipelines used in the project:

- **data_ingestion**: Loads and ingests raw data.
- **data_unit_tests** (deprecated by Great Expectations): Runs unit tests on ingested or processed data.
- **data_preprocessing**: Cleans and prepares data for modeling.
- **data_splitting**: Splits data into training, validation, and test sets.
- **model_assessment**: Evaluates trained model performance.
- **model_train**: Trains machine learning models on prepared data.
- **model_predict**: Generates predictions using trained models.
- **data_drift**: Monitors data drift in production data.

It defines `register_pipelines` to map pipeline names to `Pipeline` objects,
enabling consistent and modular execution for development, training, prediction,
assessment, and monitoring workflows in the project.
"""

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from stock_project.pipelines import (
    data_ingestion as ingestion
    ,data_unit_tests as unit_tests
    ,data_preprocessing as preprocessing
    ,data_splitting as split
    ,model_assessment as assessment
    ,model_train as train
    ,model_predict as predict
    ,data_drift as drift
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = ingestion.create_pipeline()
    unit_tests_pipeline = unit_tests.create_pipeline()
    preprocessing_pipeline = preprocessing.create_pipeline()
    splitting_pipeline = split.create_pipeline()
    assessment_pipeline = assessment.create_pipeline()
    train_pipeline = train.create_pipeline()
    prediction_pipeline = predict.create_pipeline()
    drift_pipeline = drift.create_pipeline()

    return {
        "__default__": preprocessing_pipeline + splitting_pipeline + prediction_pipeline
        ,"ingestion": ingestion_pipeline
        ,"unit_tests": unit_tests_pipeline
        ,"preprocess": preprocessing_pipeline
        ,"split": splitting_pipeline
        ,"assessment": assessment_pipeline
        ,"train": train_pipeline
        ,"production_full_train": preprocessing_pipeline + splitting_pipeline + train_pipeline
        ,"predict": prediction_pipeline
        ,"production_full_prediction": preprocessing_pipeline + splitting_pipeline + prediction_pipeline
        ,"drift": drift_pipeline
    }