"""
This is a pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import collect_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=collect_data
            ,inputs=dict(
                symbols="params:symbols"
                ,start_date="params:start_date"
            )
            ,outputs="raw_data"
            ,name="collect_data_node"
        )
    ])
