import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.stock_project.pipelines.model_predict.nodes import make_predictions  # replace with actual import path

def test_make_predictions():
    X_test = pd.DataFrame({"feature": [1, 2, 3]}, index=[10, 11, 12])
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [0, 1, 0]

    result = make_predictions(X_test, mock_pipeline)

    assert isinstance(result, pd.Series)
    assert result.name == "y_pred"
    assert result.index.equals(X_test.index)
    assert list(result) == [0, 1, 0]
    mock_pipeline.predict.assert_called_once_with(X_test)
