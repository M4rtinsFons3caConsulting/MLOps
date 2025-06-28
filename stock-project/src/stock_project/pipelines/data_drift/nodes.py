"""
Pipeline 'data_drift' node for data drift detection

Provides drift detection logic for the pipeline using NannyML and Evidently AI.

Functions called:
- `nml.UnivariateDriftCalculator.fit` and `.calculate`: Calculates univariate drift on numerical columns.
- `nml.DataReconstructionDriftCalculator.fit` and `.calculate`: Calculates reconstruction drift on numerical columns.
- `plot(kind='drift').write_html`: Generates and saves HTML drift plots for monitoring.
- `evidently.Report.run` and `.save_html`: Generates and saves an Evidently HTML drift report.

Used for monitoring data drift between reference (training) and analysis (incoming) datasets.
"""

from typing import Tuple
import pandas as pd
import nannyml as nml
from evidently.report import Report
from sklearn.impute import SimpleImputer
from evidently.metric_preset import DataDriftPreset


def apply_drift(
    data_reference: pd.DataFrame,
    data_analysis: pd.DataFrame,
    chunk_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze univariate and reconstruction drift on numerical columns using NannyML and Evidently AI.

    Returns:
        Tuple containing:
            - Univariate drift results DataFrame from NannyML
            - Reconstruction drift results DataFrame from NannyML
    """

    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    constant_threshold.thresholds(data_reference)

    numerical_columns = data_reference.select_dtypes(include='number').columns.tolist()

    # Univariate drift calculator
    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=numerical_columns,
        chunk_size=chunk_size,
        treat_as_categorical=[],
        thresholds={"jensen_shannon": constant_threshold}
    )
    univariate_calculator.fit(data_reference)

    # Reconstruction drift calculator
    reconstruction_calculator = nml.DataReconstructionDriftCalculator(
        column_names=numerical_columns,
        timestamp_column_name='date',
        chunk_size=chunk_size,
        imputer_categorical=SimpleImputer(strategy='constant', fill_value='missing'),
        imputer_continuous=SimpleImputer(strategy='median')
    )
    reconstruction_calculator.fit(data_reference)

    # Calculate univariate drift
    univariate_results = (
        univariate_calculator
        .calculate(data_analysis)
        .filter(period='analysis', column_names=numerical_columns, methods=["jensen_shannon"])
        .to_df()
    )

    # Calculate reconstruction drift
    reconstruction_results = (
        reconstruction_calculator
        .calculate(data_analysis)
        .filter(period='analysis', column_names=numerical_columns, methods=["jensen_shannon"])
        .to_df()
    )

    # Generate and save plot (side effect)
    figure = (
        univariate_calculator
        .calculate(data_analysis)
        .filter(period='analysis', column_names=numerical_columns, methods=["jensen_shannon"])
        .plot(kind='drift')
    )
    figure.write_html("data/06_reporting/drift_plot.html")

    # Generate and save Evidently report (side effect)
    drift_report = Report(
        metrics=[DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)]
    )
    drift_report.run(
        current_data=data_analysis[numerical_columns],
        reference_data=data_reference[numerical_columns]
    )
    drift_report.save_html("data/06_reporting/drift_report.html")

    return univariate_results, reconstruction_results
