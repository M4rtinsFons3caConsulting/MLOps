"""
This is a pipeline 'model_predict'
generated using Kedro 0.19.14
"""


import pandas as pd
import logging
import pickle

logger = logging.getLogger(__name__)

def make_predictions(
    X_test: pd.DataFrame
    ,pipeline: pickle
) -> pd.Series:
    """Predict using the trained pipeline.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        pipeline (pickle): Trained pipeline.

    Returns:
    --
         pd.Series: Predicted labels indexed by X_test.
    """
    # Predict
    y_pred = pipeline.predict(X_test)

    logger.info('Predictions created.')

    return pd.Series(y_pred, index=X_test.index, name="y_pred")