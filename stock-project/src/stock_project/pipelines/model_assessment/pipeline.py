"""
Pipeline 'model_assessment' for selecting the best model.

Defines a Kedro pipeline that runs the `find_champion_model` node,
which performs model selection using training data, parameters,
hyperparameters, cross-validation, and scoring metric.

Functions:
----------
create_pipeline(**kwargs) -> Pipeline
    Constructs and returns the model assessment pipeline.
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
