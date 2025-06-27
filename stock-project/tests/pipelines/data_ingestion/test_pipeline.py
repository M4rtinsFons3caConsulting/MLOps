"""
Unit tests for the data_ingestion pipeline (core coverage):

Test for `build_expectation_suite`:
1. Returns valid `ExpectationSuite` for well-formed DataFrame

Test for `to_feature_store`:
2. Inserts data into feature group

Test for `collect_yf_data`:
3. Ingests new data starting from effective date
"""

import pandas as pd
from unittest import mock
from great_expectations.core import ExpectationSuite

from src.stock_project.pipelines.data_ingestion.nodes import (
    build_expectation_suite,
    to_feature_store,
    collect_yf_data
)

### -----------  Tests for `build_expectation_suite` ----------- ###

def test_returns_valid_expectation_suite(test_df_1):

    suite = build_expectation_suite(test_df_1)

    assert isinstance(suite, ExpectationSuite)


### -----------  Tests for `to_feature_store` ----------- ###

def test_inserts_data_into_feature_group(test_df_2):
    with mock.patch(
            "src.stock_project.pipelines.data_ingestion.nodes.hopsworks.login"
        ) as mock_login:

        mock_fg = mock.Mock()
        mock_fg.insert.return_value = None
        mock_project = mock.Mock()
        mock_project.get_feature_store.return_value.get_or_create_feature_group.return_value = \
            mock_fg
        mock_login.return_value = mock_project

        to_feature_store(
            test_df_2,
            "symbol",
            "test_fg",
            1,
            {},
            update_stats=False
        )

        mock_fg.insert.assert_called_once_with(test_df_2)


### -----------  Tests for `collect_yf_data` ----------- ###

def test_ingests_new_data_from_effective_date(test_df_3, test_df_4):

    with mock.patch(
            "src.stock_project.pipelines.data_ingestion.nodes.get_previous_ingestion_date"
        ) as mock_prev_date,\
        mock.patch(
            "src.stock_project.pipelines.data_ingestion.nodes.download_yf_data"
        ) as mock_download:

        mock_prev_date.return_value = pd.Timestamp("2023-01-05")
        mock_download.return_value = test_df_4

        result = collect_yf_data(
            test_df_3,
            "AAPL",
            upload_numerical=False,
            upload_categorical=False
        )

        assert all(result["date"] >= pd.Timestamp("2023-01-01"))
