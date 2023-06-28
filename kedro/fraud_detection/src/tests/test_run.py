"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import pytest
from kedro.config import ConfigLoader
from kedro.framework.hooks import _create_hook_manager
from pathlib import Path
from kedro.framework.context import KedroContext


from src.fraud_detection.pipelines.etl_app.nodes.data_generation import (
    generate_customer_profiles_data,
    generate_terminals_data,
    generate_transactions_data,
    generate_fraud_Scenarios_data,
)

from src.fraud_detection.pipelines.etl_app.nodes.data_exploration import (
    plot_fraud_distribution,
    plot_transactions_daily_stats,
    plot_transactions_distribution,
    get_stats,
)

from src.fraud_detection.pipelines.etl_app.nodes.feature_transformation import (
    transform_datetime_features,
    transform_customer_features,
    transform_terminal_features,
)


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd()))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="fraud_detection",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# load 01_raw data
customer_profiles_data = pd.read_csv("data/01_raw/01_customers_data.csv")
terminals_data = pd.read_csv("data/01_raw/02_terminals_data.csv")
customer_profiles_table = pd.read_csv("data/01_raw/03_customers_terminals_data.csv")
transactions_data = pd.read_csv("data/01_raw/04_transactions_data.csv")
fraud_data = pd.read_csv("data/01_raw/06_fraud_transactions_data.csv")

fraud_transactions_data = pd.read_csv(
    "data/02_intermediate/01_fraud_transactions_transformed_data.csv"
)
fraud_transactions_transformed_2 = pd.read_csv(
    "data/02_intermediate/02_fraud_transactions_transformed_data.csv"
)


class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()
        assert project_context._package_name == "fraud_detection"

    def test_generate_customer_profiles_data(self, project_context):
        # Define the expected column names in the generated dataframe
        expected_columns = [
            "CUSTOMER_ID",
            "x_customer_id",
            "y_customer_id",
            "mean_amount",
            "std_amount",
            "mean_nb_tx_per_day",
        ]

        # Define the number of customers to generate
        n_customers = 100

        # Call the function to generate customer profiles data
        customer_profiles_table = generate_customer_profiles_data(n_customers)

        # Assert that the generated dataframe has the expected column names
        assert list(customer_profiles_table.columns) == expected_columns

        # Assert that the number of rows in the generated dataframe matches the number of customers
        assert len(customer_profiles_table) == n_customers

        # Assert that the generated dataframe does not contain any NaN values
        assert not customer_profiles_table.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column
        assert customer_profiles_table.dtypes["CUSTOMER_ID"] == np.int64
        assert customer_profiles_table.dtypes["x_customer_id"] == np.float64
        assert customer_profiles_table.dtypes["y_customer_id"] == np.float64
        assert customer_profiles_table.dtypes["mean_amount"] == np.float64
        assert customer_profiles_table.dtypes["std_amount"] == np.float64
        assert customer_profiles_table.dtypes["mean_nb_tx_per_day"] == np.float64

    def test_generate_terminals_data(self, project_context):
        # Define the expected column names in the generated dataframe
        expected_columns = ["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]

        # Define the number of terminals to generate
        n_terminals = 50

        # Call the function to generate terminals data
        terminal_data = generate_terminals_data(n_terminals)

        # Assert that the generated dataframe has the expected column names
        assert list(terminal_data.columns) == expected_columns

        # Assert that the number of rows in the generated dataframe matches the number of terminals
        assert len(terminal_data) == n_terminals

        # Assert that the generated dataframe does not contain any NaN values
        assert not terminal_data.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column
        assert terminal_data.dtypes["TERMINAL_ID"] == np.int64
        assert terminal_data.dtypes["x_terminal_id"] == np.float64
        assert terminal_data.dtypes["y_terminal_id"] == np.float64

    def test_plot_fraud_distribution(self, project_context):
        # Create a sample transactions dataframe
        transactions = pd.DataFrame(
            {
                "TX_FRAUD": [
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    True,
                ]
            }
        )

        # Call the function
        fig = plot_fraud_distribution(transactions)

        # Check if the figure is not None
        assert fig is not None

        # Check if the x-axis label is set correctly
        assert fig.axes[0].get_xlabel() == "Transaction Type"

        # Check if the y-axis label is set correctly
        assert fig.axes[0].get_ylabel() == "Count"

        # Check if the x-axis tick labels are set correctly
        assert fig.axes[0].get_xticklabels()[0].get_text() == "Non-Fraud"
        assert fig.axes[0].get_xticklabels()[1].get_text() == "Fraud"

        # Check if the plot title is set correctly
        assert (
            fig.axes[0].get_title()
            == "Distribution of Fraud and Non-Fraud Transactions"
        )

        # Check if the bar annotations are correct
        # assert fig.axes[0].patches[0].get_height() == 3
        # assert fig.axes[0].patches[1].get_height() == 3

        # Cleanup - close the figure to free up resources
        plt.close(fig)

    def test_plot_transactions_distribution(self, project_context):
        # Call the function with example transactions
        df = pd.read_csv("data/01_raw/04_transactions_data.csv")
        fig = plot_transactions_distribution(df)

        assert isinstance(fig, plt.Figure)

        assert len(fig.axes) == 2

        ax1 = fig.axes[0]
        assert ax1.get_title() == "Distribution of transaction amounts"
        assert ax1.get_xlabel() == "Amount"
        assert ax1.get_ylabel() == "Number of transactions"

        ax2 = fig.axes[1]
        assert ax2.get_title() == "Distribution of transaction times"
        assert ax2.get_xlabel() == "Time (days)"
        assert ax2.get_ylabel() == "Number of transactions"

        plt.close(fig)

    def test_get_stats(self, project_context):
        data = {
            "TX_TIME_DAYS": [1, 1, 2, 2, 2, 3, 3],
            "CUSTOMER_ID": [1, 2, 3, 4, 5, 6, 7],
            "TX_FRAUD": [0, 1, 0, 1, 0, 1, 0],
        }

        data = pd.DataFrame(data)

        nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day = get_stats(data)

        expected_tx_per_day = pd.Series([2, 3, 2], index=[1, 2, 3])
        assert nb_tx_per_day.equals(expected_tx_per_day)

        # Check the number of fraudulent transactions per day
        expected_fraud_per_day = pd.Series([1, 1, 1], index=[1, 2, 3])
        assert (nb_fraud_per_day == expected_fraud_per_day).all()

        # Check the number of fraudulent cards per day
        expected_fraudcard_per_day = pd.Series([1, 1, 1], index=[1, 2, 3])
        assert nb_fraudcard_per_day.equals(expected_fraudcard_per_day)

    def test_plot_transactions_daily_stats(self, project_context):
        # Create a sample transactions dataframe for testing
        transactions = pd.DataFrame(
            {
                "TX_TIME_DAYS": [1, 1, 2, 2, 2, 3, 3],
                "CUSTOMER_ID": [1, 2, 3, 4, 5, 6, 7],
                "TX_FRAUD": [0, 1, 0, 1, 0, 1, 0],
            }
        )

        result = plot_transactions_daily_stats(transactions)

        assert isinstance(result, plt.Figure)

        # Assert that the plot title is correct
        assert (
            result.axes[0].get_title()
            == "Total transactions, and number of fraudulent transactions \n and number of compromised cards per day"
        )

        assert (
            result.axes[0].get_xlabel()
            == "Number of days since beginning of data generation"
        )

        assert result.axes[0].get_ylabel() == "Number"

        # Assert that the y-axis limits are correct
        assert result.axes[0].get_ylim() == (0, 2000)

        legend_labels = [
            "# transactions per day",
            "# fraudulent txs per day",
            "# fraudulent cards per day",
        ]
        assert [
            text.get_text() for text in result.axes[0].legend_.get_texts()
        ] == legend_labels

    def test_select_terminals_within_customer_radius(self, project_context):

        # extract one row as example
        example = eval(customer_profiles_table["available_terminals"][0])
        assert len(example) > 0

        # Assert that all values in the example list are integers
        for value in example:
            assert isinstance(value, int)

        # Assert that all values in the example list are integers
        for value in customer_profiles_table["nb_terminals"]:
            assert isinstance(value, int)

    def test_generate_transactions_data(self):

        # Define the expected column names in the generated dataframe
        expected_columns = [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
        ]

        start_date = "2020-01-01"
        nb_days = 30

        # Call the function to generate terminals data
        results_td = generate_transactions_data(
            customer_profiles_table, start_date, nb_days
        )

        # Assert that the generated dataframe has the expected column names
        assert list(results_td.columns) == expected_columns

        # Assert that the generated dataframe does not contain any NaN values
        assert not results_td.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column
        assert results_td.dtypes["TX_AMOUNT"] == np.float64

    def test_generate_fraud_Scenarios_data(self):

        # load function
        fraud_scenarios = generate_fraud_Scenarios_data(
            customer_profiles_data, terminals_data, transactions_data
        )

        # Find in TX_FRAUD column a value greater than 220
        # Assuming transactions_data is your DataFrame
        index_above_220 = fraud_scenarios.loc[
            fraud_scenarios["TX_AMOUNT"] > 220, "TX_AMOUNT"
        ].idxmax()
        index_below_220 = fraud_scenarios.loc[
            fraud_scenarios["TX_AMOUNT"] <= 220, "TX_AMOUNT"
        ].idxmax()

        # Access the value in the TX_FRAUD column at the found index
        fraud_above = fraud_scenarios.loc[index_above_220, "TX_FRAUD"]
        not_fraud = fraud_scenarios.loc[index_below_220, "TX_FRAUD"]

        # Assert that the value of fraud is equal to 1 and not_fraud is equal to 0
        assert fraud_above == 1
        assert not_fraud == 0

        # assert TX_FRAUD_SCENARIO
        index_fraud_1 = fraud_scenarios.loc[
            fraud_scenarios["TX_FRAUD_SCENARIO"] == 1, "TX_FRAUD_SCENARIO"
        ].idxmax()
        fraud_1 = fraud_scenarios.loc[index_fraud_1, "TX_AMOUNT"]

        assert fraud_1 > 220

        # Assert that in TX_FRAUD_SCENARIO exist only until 3 scenarios
        assert fraud_scenarios["TX_FRAUD_SCENARIO"].max() <= 3

        assert fraud_scenarios.dtypes["TX_FRAUD_SCENARIO"] == np.int64
        assert fraud_scenarios.dtypes["TX_FRAUD"] == np.int64

        # Select only the rows where TX_FRAUD_SCENARIO different from 0
        fraud_scenarios_1 = fraud_scenarios[fraud_scenarios["TX_FRAUD_SCENARIO"] != 0]
        # Assert that the values of tx_fraud are 1
        assert fraud_scenarios_1["TX_FRAUD"].unique() == [1]

        # Select only the rows where TX_FRAUD_SCENARIO equals to 0
        fraud_scenarios_0 = fraud_scenarios[fraud_scenarios["TX_FRAUD_SCENARIO"] == 0]
        # Assert that the values of tx_fraud are 0
        assert fraud_scenarios_0["TX_FRAUD"].unique() == [0]

    def test_transform_datetime_features(self):
        # Define the expected column names in the generated dataframe
        expected_columns_tdf = [
            "TX_DURING_WEEKEND",
            "TX_DURING_NIGHT",
        ]

        # set the TX_DATETIME column as datetime
        fraud_data["TX_DATETIME"] = pd.to_datetime(fraud_data["TX_DATETIME"])

        results_tdf = transform_datetime_features(fraud_data)

        # Drop list
        drop_list = [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]

        # Drop from results_tdf_drop the columns in drop_list
        results_tdf_drop = results_tdf.drop(drop_list, axis=1)

        # Assert that the generated dataframe has the expected column names
        assert list(results_tdf_drop.columns) == expected_columns_tdf

        # Assert that the generated dataframe does not contain any NaN values
        assert not results_tdf_drop.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column created
        assert results_tdf_drop.dtypes["TX_DURING_WEEKEND"] == np.int64
        assert results_tdf_drop.dtypes["TX_DURING_NIGHT"] == np.int64

        # Assert that the values in each column are either 0 or 1
        assert set(results_tdf_drop["TX_DURING_WEEKEND"].unique()) == {0, 1}
        assert set(results_tdf_drop["TX_DURING_NIGHT"].unique()) == {0, 1}

    def test_get_customer_spending_behaviour_features(self):
        # Define the expected column names in the generated dataframe
        expected_columns_gcsbf = [
            "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",
            "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",
            "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",
        ]

        # set the TX_DATETIME column as datetime
        fraud_transactions_data["TX_DATETIME"] = pd.to_datetime(
            fraud_transactions_data["TX_DATETIME"]
        )

        # load the customer profiles table
        results_gcsbf = transform_customer_features(fraud_transactions_data)

        # Drop list
        drop_list = [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
            "TX_DURING_WEEKEND",
            "TX_DURING_NIGHT",
        ]

        # Drop from fraud_transactions_transformed_2 the columns in drop_list
        fraud_transactions_transformed_2 = results_gcsbf.drop(drop_list, axis=1)

        # Assert that the generated dataframe has the expected column names
        assert list(fraud_transactions_transformed_2.columns) == expected_columns_gcsbf

        # Assert that the generated dataframe does not contain any NaN values
        assert not fraud_transactions_transformed_2.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column created
        assert (
            fraud_transactions_transformed_2.dtypes["CUSTOMER_ID_NB_TX_1DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_2.dtypes[
                "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW"
            ]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_2.dtypes["CUSTOMER_ID_NB_TX_7DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_2.dtypes[
                "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW"
            ]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_2.dtypes["CUSTOMER_ID_NB_TX_30DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_2.dtypes[
                "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"
            ]
            == np.float64
        )

    def test_transform_terminal_features(self):
        # Define the expected column names in the generated dataframe
        expected_columns_gcsbf = [
            "TERMINAL_ID_NB_TX_1DAY_WINDOW",
            "TERMINAL_ID_RISK_1DAY_WINDOW",
            "TERMINAL_ID_NB_TX_7DAY_WINDOW",
            "TERMINAL_ID_RISK_7DAY_WINDOW",
            "TERMINAL_ID_NB_TX_30DAY_WINDOW",
            "TERMINAL_ID_RISK_30DAY_WINDOW",
        ]

        # Drop list
        drop_list = [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
            "TX_DURING_WEEKEND",
            "TX_DURING_NIGHT",
            "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",
            "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",
            "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
            "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",
        ]

        # set the TX_DATETIME column as datetime
        fraud_transactions_transformed_2["TX_DATETIME"] = pd.to_datetime(
            fraud_transactions_transformed_2["TX_DATETIME"]
        )

        # load the customer profiles table
        results_ttf = transform_terminal_features(fraud_transactions_transformed_2)

        # Drop from fraud_transactions_transformed_3 the columns in drop_list
        fraud_transactions_transformed_3 = results_ttf.drop(drop_list, axis=1)

        # Assert that the generated dataframe has the expected column names
        assert list(fraud_transactions_transformed_3.columns) == expected_columns_gcsbf

        # Assert that the generated dataframe does not contain any NaN values
        assert not fraud_transactions_transformed_3.isnull().values.any()

        # Assert that the generated dataframe has the correct data types for each column created
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_NB_TX_1DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_RISK_1DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_NB_TX_7DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_RISK_7DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_NB_TX_30DAY_WINDOW"]
            == np.float64
        )
        assert (
            fraud_transactions_transformed_3.dtypes["TERMINAL_ID_RISK_30DAY_WINDOW"]
            == np.float64
        )
