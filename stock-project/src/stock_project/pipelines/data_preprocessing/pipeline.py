"""
Pipeline 'data_preprocessing' for preprocessing data.

This pipeline prepares model input by performing feature engineering, target creation,
data widening, and missing value handling on raw OHLCV data.

Inputs:
- raw_data: Raw OHLCV market data.
- params:is_to_feature_store: Boolean flag to enable feature store persistence.

Outputs:
- preprocessed_data: Final DataFrame ready for modeling, with features and target labels.
- feature_store_versions: Dictionary tracking feature store version numbers.
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
