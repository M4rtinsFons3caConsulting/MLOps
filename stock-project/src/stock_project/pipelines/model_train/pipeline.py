"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_train
            ,inputs=[
                "X_train"
                ,"y_train"
                ,"champion_model"
            ]
            ,outputs=[
                "production_model"
                ,"shap_plot"
            ]
            ,name="model_train_node"
        )
    ])
