"""

Pipeline 'data_splitting' for data splitting into training, and hold-out.


Defines the pipeline to split preprocessed data into training and testing sets
using the split_data function from the nodes module.

Pipeline:
- data_splitting_node: Executes split_data with inputs 'preprocessed_data' and 'test_size' parameter,
  outputs training and testing features and labels.
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data
            ,inputs=[
                "preprocessed_data"
                ,"params:test_size"
            ]
            ,outputs=[
                "X_train"
                ,"X_test"
                ,"y_train"
                ,"y_test"
            ]
            ,name="data_splitting_node"
        )
    ])
