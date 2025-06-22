"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from stock_project.pipelines import (
    data_ingestion as ingestion
    ,data_preprocessing as preprocessing
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = ingestion.create_pipeline()
    preprocessing_pipeline = preprocessing.create_pipeline()

    return {
        "ingestion": ingestion_pipeline
        ,"preprocess": preprocessing_pipeline
    }