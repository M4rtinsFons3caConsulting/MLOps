"""
Pipeline 'model_predict' for generating predictions using a trained model.

Creates a Kedro pipeline with one node that:
- Takes test features ("X_test") and the production model ("production_model") as inputs.
- Applies the `make_predictions` function to produce predictions.
- Outputs the predictions as "predictions".

Generated with Kedro 0.19.14.
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
