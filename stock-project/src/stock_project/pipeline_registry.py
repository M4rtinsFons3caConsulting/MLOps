"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from stock_project.pipelines import (
    data_ingestion as ingestion
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
    preprocessing_pipeline = preprocessing.create_pipeline()
    splitting_pipeline = split.create_pipeline()
    assessment_pipeline = assessment.create_pipeline()
    train_pipeline = train.create_pipeline()
    prediction_pipeline = predict.create_pipeline()
    drift_pipeline = drift.create_pipeline()

    return {
        "ingestion": ingestion_pipeline
        ,"preprocess": preprocessing_pipeline
        ,"split": splitting_pipeline
        ,"assessment": assessment_pipeline
        ,"train": train_pipeline
        ,"predict": prediction_pipeline
        ,"drift": drift_pipeline
    }