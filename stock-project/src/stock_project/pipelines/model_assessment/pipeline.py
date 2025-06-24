"""
This is a pipeline 'model_assessment'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import find_champion_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=find_champion_model
            ,inputs=[
                "X_train"
                ,"y_train"
                ,"params:parameters"
                ,"params:hyperparameters"
                ,"params:n_trials"
                ,"params:scoring_metric"
                ,"params:cv_args"
                ,"params:kwargs"
            ]
            ,outputs="champion_model"
            ,name="model_assessment_node"
        )
    ])
