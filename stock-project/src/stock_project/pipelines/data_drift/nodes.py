"""
This is a pipeline 'data_drift'
generated using Kedro 0.19.14
"""

from typing import Tuple
import pandas as pd
import nannyml as nml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
# from evidently.metrics import DataDriftPreset

def apply_drift(
    data_reference: pd.DataFrame,
    data_analysis: pd.DataFrame
) -> Tuple[pd.DataFrame, "plotly.graph_objs.Figure", Report]:
    """
    Analyze univariate drift on numerical columns using NannyML and Evidently AI.

    Args:
        data_reference (pd.DataFrame): Historical or training data.
        data_analysis (pd.DataFrame): New or production data.

    Returns:
        Tuple containing:
            - Drift results DataFrame from NannyML
            - NannyML drift plot (plotly Figure)
            - Evidently drift report object
    """
    # Define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    constant_threshold.thresholds(data_reference)

    numerical_columns = data_reference.select_dtypes(include='number').columns.tolist()

    # NannyML univariate drift
    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=numerical_columns,
        chunk_size=50,
        treat_as_categorical=[],
        thresholds={"jensen_shannon":constant_threshold}
    )

    univariate_calculator.fit(data_reference)

    # Calculate drift
    results = (
        univariate_calculator
        .calculate(data_analysis)
        .filter(
            period='analysis'
            ,column_names=numerical_columns
            ,methods=["jensen_shannon"]
        )
        .to_df()
    )

    # Generate drift plot
    figure = (
        univariate_calculator
        .calculate(data_analysis)
        .filter(
            period='analysis'
            ,column_names=numerical_columns
            ,methods=["jensen_shannon"]
        )
        .plot(kind='drift')
    )
    figure.write_html("data/06_reporting/drift_plot.html")

    # Evidently report
    drift_report = Report(
        metrics=[
            DataDriftPreset(
                cat_stattest='ks'
                ,stattest_threshold=0.05
            )
        ]
    )
    drift_report.run(
        current_data=data_analysis[numerical_columns],
        reference_data=data_reference[numerical_columns]
    )
    drift_report.save_html("data/06_reporting/drift_report.html")

    return results
