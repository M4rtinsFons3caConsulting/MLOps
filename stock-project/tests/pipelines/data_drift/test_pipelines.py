import pandas as pd
import numpy as np
import pytest
from src.stock_project.pipelines.data_drift.nodes import apply_drift

def test_apply_drift():
    data_reference = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'feature1': np.random.normal(size=100),
        'feature2': np.random.uniform(size=100),
        'feature3': np.random.randint(0, 10, size=100)
    })
    data_analysis = pd.DataFrame({
        'date': pd.date_range('2023-04-11', periods=50),
        'feature1': np.random.normal(size=50),
        'feature2': np.random.uniform(size=50),
        'feature3': np.random.randint(0, 10, size=50)
    })

    univariate_results, reconstruction_results = apply_drift(data_reference, data_analysis, 5)

    assert isinstance(univariate_results, pd.DataFrame), "Univariate results should be a DataFrame"
    assert isinstance(reconstruction_results, pd.DataFrame), "Reconstruction results should be a DataFrame"
