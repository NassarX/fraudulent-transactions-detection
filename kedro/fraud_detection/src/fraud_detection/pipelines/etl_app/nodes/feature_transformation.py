import datetime
import os

import pandas as pd


def transform_datetime_features(transactions_data: pd.DataFrame) -> pd.DataFrame:
    """
    Date/Time transformation involves the date/time variable, and consists in creating binary features that
    characterize potentially relevant periods. We will create two such features. The first one will characterize
    whether a transaction occurs during a weekday or during the weekend. The second will characterize whether a
    transaction occurs during the day or the night. These features can be useful since it has been observed in
    real-world datasets that fraudulent patterns differ between weekdays and weekends, and between the day and night.

    Args:
        transactions_data: a dataframe with transactions data
    Returns:
        transactions_data: a dataframe with updated transactions data
    """

    def is_weekend(tx_datetime):
        # Transform date into weekday (0 is Monday, 6 is Sunday)
        weekday = tx_datetime.weekday()
        # Binary value: 0 if weekday, 1 if weekend
        weekend = weekday >= 5
        return int(weekend)

    def is_night(tx_datetime):
        # Get the hour of the transaction
        tx_hour = tx_datetime.hour
        # Binary value: 1 if hour is less than or equal to 6, and 0 otherwise
        night = tx_hour <= 6
        return int(night)

    # Create new columns based on the existing 'TX_DATETIME' column
    transactions_data['TX_DURING_WEEKEND'] = transactions_data['TX_DATETIME'].apply(is_weekend)
    transactions_data['TX_DURING_NIGHT'] = transactions_data['TX_DATETIME'].apply(is_night)

    return transactions_data


def transform_customer_features(transactions_data: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Transformation involves the customer ID and consists in creating features that characterize the customer
    spending behaviors. We will follow the RFM (Recency, Frequency, Monetary value) framework proposed in {
    cite} VANVLASSELAER201538, and keep track of the average spending amount and number of transactions for each
    customer and for three window sizes. This will lead to the creation of six new features.

    Args:
        transactions_data: a dataframe with transactions data
    Returns:
        transactions_data: a dataframe with updated transactions data
    """

    def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
        # Let us first order transactions chronologically
        customer_transactions = customer_transactions.sort_values('TX_DATETIME')

        # The transaction date and time is set as the index, which will allow the use of the rolling function
        customer_transactions.index = customer_transactions.TX_DATETIME

        # For each window size
        for window_size in windows_size_in_days:
            # Convert window size from days to periods
            window_size_alias = f"{window_size}D"

            # Compute the sum of the transaction amounts and the number of transactions for the given window size
            SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(window=window_size_alias).sum()
            NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(window=window_size_alias).count()

            # Compute the average transaction amount for the given window size
            # NB_TX_WINDOW is always >0 since current transaction is always included
            AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

            # Save feature values
            customer_transactions['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
            customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(
                AVG_AMOUNT_TX_WINDOW)

        # Reindex according to transaction IDs
        customer_transactions.index = customer_transactions.TRANSACTION_ID

        # And return the dataframe with the new features
        return customer_transactions

    transactions_data = transactions_data.groupby('CUSTOMER_ID').apply(
        lambda x: get_customer_spending_behaviour_features(x, [1, 7, 30]))
    transactions_data = transactions_data.sort_values('TX_DATETIME').reset_index(drop=True)

    return transactions_data


def transform_terminal_features(transactions_data: pd.DataFrame) -> pd.DataFrame:
    """
    Terminal Transformation involves the terminal ID and consists in creating new features that characterize the
    'risk' associated with the terminal. The risk will be defined as the average number of frauds that were observed
    on the terminal for three window sizes. This will lead to the creation of three new features.

    Args:
        transactions_data: a dataframe with transactions data
    Returns:
        transactions_data: a dataframe with updated transactions data
    """

    def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30],
                                      feature="TERMINAL_ID"):
        terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')

        terminal_transactions.index = terminal_transactions.TX_DATETIME

        NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(window=delay_period, min_periods=delay_period).sum()
        NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(window=delay_period, min_periods=delay_period).count()

        for window_size in windows_size_in_days:
            # Convert window size from days to periods
            window_size_alias = f"{window_size + delay_period}D"

            NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(
                window=window_size_alias).sum()
            NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(
                window=window_size_alias).count()

            NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
            NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

            RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

            terminal_transactions[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
            terminal_transactions[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)

        terminal_transactions.index = terminal_transactions.TRANSACTION_ID

        # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
        terminal_transactions.fillna(0, inplace=True)

        return terminal_transactions

    transactions_data = transactions_data.groupby('TERMINAL_ID').apply(
        lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30],
                                                feature="TERMINAL_ID"))
    transactions_data = transactions_data.sort_values('TX_DATETIME').reset_index(drop=True)

    return transactions_data
