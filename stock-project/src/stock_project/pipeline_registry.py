"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from stock_project.pipelines import (
    data_ingestion as ingestion
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = ingestion.create_pipeline()

    return {
        "ingestion": ingestion_pipeline,
    }