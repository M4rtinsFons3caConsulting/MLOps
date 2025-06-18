"""
This is a 'data_engineering'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import create_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_features
            ,inputs="ingested_data"
            ,outputs="engineered_data"
            ,name="data_engineering_node"
        )
    ])
