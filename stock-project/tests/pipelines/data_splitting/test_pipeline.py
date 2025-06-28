import pandas as pd
from src.stock_project.pipelines.data_splitting.nodes import split_data

def test_split_data_shapes_and_types():
    # Create dummy data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'feature1': range(10),
        'label': range(10)
    })
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.3)
    
    # Check types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Check shapes
    expected_test_size = int(10 * 0.3)
    expected_train_size = 10 - expected_test_size

    assert X_train.shape[0] == expected_train_size
    assert X_test.shape[0] == expected_test_size
    assert y_train.shape[0] == expected_train_size
    assert y_test.shape[0] == expected_test_size

