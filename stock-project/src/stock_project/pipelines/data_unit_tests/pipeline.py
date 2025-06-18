"""
This is a pipeline 'data_unit_tests'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=test_data,
            inputs="ingested_data",
            outputs= "reporting_tests",
            name="data_unit_tests",
        )
    ])
