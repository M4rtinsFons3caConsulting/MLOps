"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import apply_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_drift,
            inputs=[
                "X_train"
                ,"X_test"
                ,"params:chunk_size"
            ],
            outputs=[
                "univariate_drift_results"
                ,"multivariate_drift_results"
            ],
            name="data_drift_node"
        )
    ])
