"""
This is a 'data_preprocessing'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import perform_feature_engineering, create_target, widden_df, prepare_model_input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=perform_feature_engineering
            ,inputs="ingested_data"
            ,outputs="engineered_data"
            ,name="data_engineering_node"
        ), node(
            func=create_target
            ,inputs=["engineered_data", "params:prediction_horizon", "params:threshold"]
            ,outputs="labels"
            ,name="create_target_node"
        ), node(
            func=widden_df
            ,inputs="engineered_data"
            ,outputs="widden_data"
            ,name="widden_df_node"
        ), node(
            func=prepare_model_input
            ,inputs=["widden_data", "labels"]
            ,outputs="preprocessed_data"
            ,name="prepare_model_input_node"
        )
    ])
