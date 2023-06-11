import os
from typing import Any, Callable, Dict, Tuple
import pandas as pd
from datetime import timedelta, datetime as dt

from pandas import DataFrame


def split_transaction_data_into_batches(transactions_data: pd.DataFrame, start_date: str) -> Dict[str, Any]:
    """Split transaction data into batches by day.
        params:
            transactions_data: DataFrame containing transactions data
            start_date: Date from which to start splitting data into batches
        return:
            Dictionary containing batches of transactions data
    """
    start_date = dt.strptime(start_date, "%Y-%m-%d")
    batches = {}
    for day in range(transactions_data.TX_TIME_DAYS.max() + 1):
        transactions_per_day = transactions_data[transactions_data.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        date = start_date + timedelta(days=day)
        filename = date.strftime("%Y-%m-%d") + '.csv'
        batches[filename] = transactions_per_day

    return batches


def process_daily_transaction_data(partitioned_input: Dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """Concatenate input partitions into one pandas DataFrame.

    Args:
        partitioned_input: A dictionary with partition ids as keys and load functions as values.

    Returns:
        Pandas DataFrame representing a concatenation of all loaded partitions.
    """
    merged_df = pd.DataFrame()

    for partition_id, partition_load_func in sorted(partitioned_input.items()):
        if partition_id == ".gitkeep":
            continue

        partition_data = partition_load_func()  # load actual partition data
        merged_df = pd.concat([merged_df, partition_data], ignore_index=True, sort=True)  # concat with existing result

    return merged_df


def split_train_test_data(processed_df: pd.DataFrame, start_date_train: str, delta_train=7, delta_delay=7,
                          delta_test=7) -> Tuple[Any, Any, Any, Any]:
    """Split processed dataset into train and test sets

    Args:
        processed_df (pd.DataFrame): Dataframe containing the processed transaction dataset
        start_date_train (str): start date for train data in 'YYYY-MM-DD' format
        delta_train (int): period for training
        delta_delay (int): delay period
        delta_test (int): period for testing

    Returns:
        Tuple of pandas dataframes: x_train, y_train, x_test, y_test
    """
    # Convert the string to a datetime object
    start_date_train = dt.strptime(start_date_train, '%Y-%m-%d')

    # Perform chronological train test split (80:20) i.e. 8 weeks:2 weeks

    # Get the training set data
    train_df = processed_df[
        (processed_df.TX_DATETIME >= start_date_train) &
        (processed_df.TX_DATETIME < start_date_train + timedelta(days=delta_train))
    ]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day - delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = processed_df[
            processed_df.TX_TIME_DAYS == start_tx_time_days_training + delta_train + delta_delay + day
        ]

        # Compromised cards from that test day, minus the delay period,
        # are added to the pool of known defrauded customers
        test_df_day_delay_period = processed_df[
            processed_df.TX_TIME_DAYS == start_tx_time_days_training + delta_train + day - 1
        ]

        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Split train data into features (x_train) and labels (y_train)
    x_train = train_df.drop(columns=['TX_FRAUD'])
    y_train = train_df['TX_FRAUD']

    # Split test data into features (x_test) and labels (y_test)
    x_test = test_df.drop(columns=['TX_FRAUD'])
    y_test = test_df['TX_FRAUD']

    return x_train, y_train, x_test, y_test

