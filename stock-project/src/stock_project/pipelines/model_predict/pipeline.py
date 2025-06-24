"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import make_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=make_predictions
            ,inputs=[
                "X_test"
                ,"production_model"
            ]
            ,outputs="predictions"
            ,name="model_predict_node"
        )
    ])
