"""
Pipeline 'data_ingestion' node for data ingestion and wrangling.

Module for ingesting, validating, profiling, and storing stock market data.

This module provides functions to:
- Collect historical stock data from yfinance with incremental ingestion.
- Automatically generate Great Expectations expectation suites using ydata_profiling.
- Validate data against expectation suites.
- Store validated data into a feature store using hopsworks.
- Track ingestion metadata and feature store versioning with Kedro and MLflow.

Dependencies:
- pandas, yfinance, kedro, mlflow, great_expectations, ydata_profiling, hopsworks

Key Functions:
- collect_yf_data: Ingests stock data incrementally, profiles, validates, optionally stores features.
- build_expectation_suite: Creates an expectation suite from a DataFrame using profiling.
- to_feature_store: Validates and writes data to the feature store with metadata and statistics.

"""

# STL
import os
import logging
from pathlib import Path
from typing import List, Any, Dict, Tuple
from datetime import datetime, timedelta, date

# Ingestion
import pandas as pd
import yfinance as yf

# Pipeline
from kedro.framework.session import KedroSession
from kedro.io.core import DatasetError
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Logging
import mlflow

# Expectations
import great_expectations as gx
from great_expectations.data_context import DataContext
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.core.batch import RuntimeBatchRequest
from ydata_profiling import ProfileReport
from ydata_profiling.expectations_report import ExpectationsReport
from .auto_expectations import ExpectationsReportV3
import hopsworks

# Configs
conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

# Logger
logger = logging.getLogger(__name__)


def build_expectation_suite(
    df
    , suite_name="stock_auto_suite"
    , datasource_name="stock_datasource"
    , data_asset_name="stock_asset"
):
    """
    Build and save an expectation suite using ydata_profiling, with optional customization.

    Args:
        df (pd.DataFrame): The DataFrame to profile and validate.
        suite_name (str): Name of the expectation suite.
        datasource_name (str): Name of the GE datasource.
        data_asset_name (str): Name of the GE data asset.

    Returns:
        ExpectationSuite: The generated expectation suite.
    """
    # Get GE context
    context_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../gx"))
    context = DataContext(context_root_dir=context_root_dir)

    # Set up or retrieve datasource
    try:
        datasource = context.add_datasource(
            name=datasource_name,
            class_name="Datasource",
            execution_engine={
                "class_name": "PandasExecutionEngine"
            },
            data_connectors={
                "default_runtime_data_connector_name": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"],
                }
            },
        )
    except:
        datasource = context.datasources[datasource_name]

    # Set up or retrieve asset
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name=data_asset_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_id"},
    )

    # Profile the data using ydata_profiling
    profile = ProfileReport(df, title="Auto Profiling Report", minimal=True)
    ExpectationsReport.to_expectation_suite = ExpectationsReportV3.to_expectation_suite

    auto_suite = profile.to_expectation_suite(
        datasource_name=datasource_name,
        data_asset_name=data_asset_name,
        suite_name=suite_name,
        data_context=context,
        run_validation=False,
        dataframe=df
    )

    # modify some expectations
    exp_columns = {'close', 'high', 'low', 'open', 'volume'}
    for exp in auto_suite.expectations:
        if (
            exp.expectation_type == "expect_column_values_to_be_between"
            and exp.kwargs.get("column") in exp_columns
        ):
            exp.kwargs['min_value'] = 0

    # Save the expectation suite
    context.save_expectation_suite(expectation_suite=auto_suite)

    return auto_suite


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # Upload data.
    object_feature_group.insert(
        features=data,
        write_options={
            "wait_for_job": True,
            "schema_evolution": True
        }
    )

    # Add feature descriptions.
    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def collect_yf_data(
    symbols: List[str],
    user_start_date: str,
    last_ingestion_date=None,
    is_to_feature_store: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Ingests yfinance data starting from max(user_start_date, last_ingestion_date)
    up to yfinance's default end (i.e. latest available).
    Returns:
        - DataFrame with columns: ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        - The latest date in the returned data as new last_ingestion_date
    """

    ### --------------------------------- DATA INGESTION ---------------------------------###
    date_format = "%Y-%m-%d"
    project_path = Path(__file__).resolve().parents[4]

    # Inject previous context
    with KedroSession.create(project_path=project_path) as session:
        catalog = session.load_context().catalog
        
        # Get previous ingestion date
        try:
            last_ingestion_date = catalog.load("last_ingestion_date")
        except DatasetError:
            last_ingestion_date = user_start_date

        # Try to load the data
        try:
            last_data_ingested = catalog.load("raw_data")
        except DatasetError:
            last_data_ingested = None

        # Try to load the feature store versioning
        try:
            versions = catalog.load("feature_store_versions")
        except DatasetError:
            versions = {}

    with mlflow.start_run(run_name="data_ingestion_yfinance", nested=True):
        mlflow.log_param("symbols", symbols)
        mlflow.log_param("last_ingestion_date", last_ingestion_date)

        # Catch all
        concatenate = False
        if isinstance(last_data_ingested, pd.DataFrame):
            concatenate = True

        if not last_ingestion_date:
            effective_start = datetime.strptime(user_start_date, date_format).date()
        else:
            # Infer effective start
            effective_start = max(
                datetime.strptime(user_start_date, date_format).date(),
                datetime.strptime(last_ingestion_date, date_format).date()
            )

            effective_start = effective_start + timedelta(days=1)
        mlflow.log_param("effective_start", str(effective_start))
        
        def last_business_day(ref_date: date) -> date:
            # Get last 5 business days up to ref_date
            bdays = pd.bdate_range(end=ref_date, periods=1)
            return bdays[-1].date()

        if effective_start >= last_business_day(date.today()):
            # no need to download
            logger.info(f"No data returned starting from {effective_start}")
            
            return last_data_ingested, last_ingestion_date, versions
            
        else:
            logger.info(f"Starting data ingestion for symbols: {symbols}")
            logger.info(f"Downloading data from yfinance starting from {effective_start}")

            # Ingestion
            data = yf.download(
                tickers=symbols,
                start=effective_start.strftime(date_format),
                group_by="ticker",
                auto_adjust=False,
                progress=False
            )

            mlflow.log_metric("num_new_records", len(data))
            logger.info(f"Finished downloading data. Number of records: {len(data)}")

            # Reformat into long format with ticker column
            data = (
                data
                .stack(level=0)
                .rename_axis(['date', 'symbol'])
                .reset_index()
                .rename(columns=str.lower)
            )

            logger.info(f"Reformatted data to long format. Shape: {data.shape}")

            # Select only ohlcv
            data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            # Infer latest available date from actual data
            latest_available_date = data['date'].max().date().strftime(date_format)
            
            # Append the data as new if data is present
            if concatenate:
                logger.info(f"Inserted {len(data)} columns.")
                data = pd.concat(
                    [last_data_ingested
                    , data]
                , axis=0
                )

        logger.info(f"Dataset currently contains {len(data.columns)} columns.")

        ### --------------------------------- SETTING EXPECTATIONS ---------------------------------###
        # Getting the features
        numerical_features = data.select_dtypes(exclude=['object','string','category']).columns.tolist()
        numerical_features.remove('date')
        categorical_features = data.select_dtypes(include=['object','string','category']).columns.tolist()
        
        # Setting the feature store
        data_numeric = data.drop(
            columns=categorical_features
        )
        data_categorical = data.drop(
            columns=numerical_features
        )

        # Building expectations suites
        validation_expectation_suite_numerical = build_expectation_suite(
            df=data_numeric
            ,suite_name="numerical_expectations"
            ,datasource_name="numerical_features"
            ,data_asset_name="numerical_features_asset"
        )
        validation_expectation_suite_categorical = build_expectation_suite(
            df=data_categorical
            ,suite_name="categorical_expectations"
            ,datasource_name="categorical_features"
            ,data_asset_name="categorical_features_asset"
        )

        context_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../gx"))
        mlflow.log_artifacts(context_root_dir, artifact_path="great_expectations")
        
        numerical_feature_descriptions = []
        categorical_feature_descriptions = []

        if is_to_feature_store:
            logger.info("Retrieving feature store versions...")

            # Initialize versions if missing
            if "numerical_features" not in versions:
                versions["numerical_features"] = 0
            if "categorical_features" not in versions:
                versions["categorical_features"] = 0
                
            logger.info("Feature store versions retrieved.")

            logger.info("Uploading numerical features to feature store...")

            # Update feature store version
            versions["numerical_features"] += 1
            # Add features to feature store
            object_fs_numerical_features = to_feature_store(
                data_numeric
                ,"numerical_features"
                ,versions["numerical_features"]
                ,"Numerical Features"
                ,numerical_feature_descriptions
                ,validation_expectation_suite_numerical
                ,credentials["feature_store"]
            )

            logger.info("Numerical features upload complete.")

            logger.info("Uploading categorical features to feature store...")

            # Update feature store version
            versions["categorical_features"] += 1
            # Add features to feature store
            object_fs_categorical_features = to_feature_store(
                data_categorical
                ,"categorical_features"
                ,versions["categorical_features"]
                ,"Categorical Features"
                ,categorical_feature_descriptions
                ,validation_expectation_suite_categorical
                ,credentials["feature_store"]
            )

            logger.info("Categorical features upload complete.")

        logger.info("Data ingestion complete. Returning dataset and latest available date.")
        
        return data, latest_available_date, versions
