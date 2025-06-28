"""
Pipeline 'data_drift' for data drift detection.

This pipeline detects data drift using training and test feature sets.
It applies univariate and multivariate drift checks to monitor stability
and data quality in production pipelines.

"""

from kedro.pipeline import node, Pipeline, pipeline
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
