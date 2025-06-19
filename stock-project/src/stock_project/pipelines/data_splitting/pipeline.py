"""
This is a boilerplate pipeline 'data_splitting'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data
            ,inputs=["preprocessed_data", "params:test_size", "params:random_state"]
            ,outputs=["stock_train", "stock_test"]
            ,name="data_splitting_node"
        )
    ])
