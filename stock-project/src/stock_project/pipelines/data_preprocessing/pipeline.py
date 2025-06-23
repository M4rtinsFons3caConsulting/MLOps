"""
This is a 'data_preprocessing'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import perform_feature_engineering, create_target, widden_df, prepare_model_input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_model_input
            ,inputs=[
                "raw_data"
                ,"params:is_to_feature_store"
            ]
            ,outputs=[
                "preprocessed_data"
                ,"feature_store_versions"
            ]
            ,name="prepare_model_input_node"
        )
    ])
