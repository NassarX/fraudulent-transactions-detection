import numpy as np
import pandas as pd
import random
import ast


def generate_customer_profiles_data(n_customers: int, random_state=0) -> pd.DataFrame:
    """Generate customer profiles data.
    params:
        n_customers: number of customers to generate
        random_state: random state for reproducibility
    returns:
        customer_profiles_table: dataframe containing customer profiles
    """

    np.random.seed(random_state)
    customer_properties = []

    # Generate customer properties from random distributions
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        # Generate random distributions for the customer properties
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2
        mean_nb_tx_per_day = np.random.uniform(0, 4)

        # Append the customer properties to the list
        customer_properties.append([customer_id,
                                    x_customer_id, y_customer_id,
                                    mean_amount, std_amount,
                                    mean_nb_tx_per_day])

    # Create a dataframe from the list of customer properties
    customer_profiles_table = pd.DataFrame(customer_properties, columns=[
        'CUSTOMER_ID',
        'x_customer_id',
        'y_customer_id',
        'mean_amount',
        'std_amount',
        'mean_nb_tx_per_day'])
    return customer_profiles_table


def generate_terminals_data(n_terminals: int, random_state=1) -> pd.DataFrame:
    """Generate terminals data.
        params:
            n_terminals: number of terminals to generate
            random_state: random seed

        returns:
            terminal_profiles_table: a dataframe with terminal profiles data
    """
    np.random.seed(random_state)

    terminal_properties = []

    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        terminal_properties.append([terminal_id, x_terminal_id, y_terminal_id])

    # Create a dataframe from the list of terminal properties
    terminal_data = pd.DataFrame(terminal_properties, columns=[
        'TERMINAL_ID',
        'x_terminal_id',
        'y_terminal_id'])

    return terminal_data


def select_terminals_within_customer_radius(customer_profile: dict, terminals_data: pd.DataFrame,
                                            radius: float) -> list:
    """Calculate the distance between each customer and each terminal and update the customer profiles dataframe.
        params:
            customer_profile: a dictionary with customer profile data
            terminals_data: a dataframe with terminal profiles data
            radius: radius of the circle around the customer

        returns:
            updated_customer_profiles_data: the updated customer profiles dataframe
    """
    # Use numpy arrays in the following to speed up computations
    x_y_terminals = terminals_data[['x_terminal_id', 'y_terminal_id']].values.astype(float)

    # Get customer coordinates
    x_y_customer = np.array([customer_profile['x_customer_id'], customer_profile['y_customer_id']], dtype=float)

    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    # Sum along rows and compute squared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get the indices of terminals which are at a distance less than radius
    available_terminals = list(np.where(dist_x_y < radius)[0])

    return available_terminals


def map_terminals_to_customers(customers_data: pd.DataFrame, terminals_data: pd.DataFrame,
                               radius: float) -> pd.DataFrame:
    """Map terminals to customers.
        params:
            customers_data: a dataframe with customer profiles data
            terminals_data: a dataframe with terminal profiles data
            radius: radius of the circle around the customer

        returns:
            updated_customer_profiles_data: the updated customer profiles dataframe
    """

    # Update the 'available_terminals' column using the custom function
    customers_data['available_terminals'] = customers_data.apply(
        lambda customer: select_terminals_within_customer_radius(customer, terminals_data, radius), axis=1
    )
    customers_data['nb_terminals'] = customers_data.available_terminals.apply(len)

    return customers_data


def generate_transactions_data(customers_terminals_data: pd.DataFrame, start_date: str, nb_days: int) -> pd.DataFrame:
    """Generate transactions data.
        params:
            customers_terminals_data: a dataframe with customers and terminals data
            start_date: start date of the transactions
            nb_days: number of days to generate transactions for

        returns:
            transactions_data: a dataframe with transactions data
    """
    transactions_data = pd.DataFrame(columns=[
        'TX_DATETIME',
        'CUSTOMER_ID',
        'TERMINAL_ID',
        'TX_AMOUNT',
        'TX_TIME_SECONDS',
        'TX_TIME_DAYS'])

    for _, customer_profile in customers_terminals_data.groupby('CUSTOMER_ID'):
        customer_id = customer_profile['CUSTOMER_ID'].values[0]
        mean_amount = customer_profile['mean_amount'].values[0]
        std_amount = customer_profile['std_amount'].values[0]
        mean_nb_tx_per_day = customer_profile['mean_nb_tx_per_day'].values[0]
        nb_terminals = customer_profile['nb_terminals'].values[0]
        available_terminals = ast.literal_eval(customer_profile['available_terminals'].values[0])
        # available_terminals = customer_profile['available_terminals'].values[0]

        if nb_terminals == 0:
            continue  # Skip generating transactions if no available terminals

        random.seed(int(customer_id))
        np.random.seed(int(customer_id))

        for day in range(nb_days):
            # Random number of transactions for that day
            nb_tx = np.random.poisson(mean_nb_tx_per_day)

            if nb_tx > 0:
                for tx in range(nb_tx):
                    # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that
                    # most transactions occur during the day.
                    time_tx = int(np.random.normal(86400 / 2, 20000))

                    # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                    if 0 < time_tx < 86400:

                        # Amount of transaction: normal distribution around mean_amount, std std_amount
                        amount = np.random.normal(mean_amount, std_amount)

                        # If amount is negative, let us draw another one from the distribution
                        if amount < 0:
                            amount = np.random.uniform(0, mean_amount * 2)

                        # Round amount to 2 decimals
                        amount = np.round(amount, decimals=2)

                        if nb_terminals > 0:
                            terminal_id = random.choice(available_terminals)

                            tx_time_seconds = time_tx + day * 86400
                            tx_time_days = day

                            transaction = pd.DataFrame({
                                'TX_DATETIME': pd.to_datetime(tx_time_seconds, unit='s', origin=start_date),
                                'CUSTOMER_ID': customer_id,
                                'TERMINAL_ID': terminal_id,
                                'TX_AMOUNT': amount,
                                'TX_TIME_SECONDS': tx_time_seconds,
                                'TX_TIME_DAYS': tx_time_days
                            }, index=[0])

                            transactions_data = pd.concat([transactions_data, transaction], ignore_index=True)

    transactions_data = transactions_data[
        ['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]

    # Sort transactions chronologically
    transactions_data = transactions_data.sort_values('TX_DATETIME')

    # Reset indices, starting from 0
    transactions_data.reset_index(inplace=True, drop=True)
    transactions_data.reset_index(inplace=True)

    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_data.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)

    return transactions_data


def generate_fraud_Scenarios_data(customer_profiles_data: pd.DataFrame, terminals_data: pd.DataFrame,
                                  transactions_data: pd.DataFrame) -> pd.DataFrame:
    """Generate fraud scenarios data.
        params:
            customer_profiles_data: a dataframe with customer profiles data
            terminals_data: a dataframe with terminal profiles data
            transactions_data: a dataframe with transactions data
        returns:
            transactions_data: a dataframe with transactions data
    """
    # By default, all transactions are genuine
    transactions_data['TX_FRAUD'] = 0
    transactions_data['TX_FRAUD_SCENARIO'] = 0

    # Scenario 1
    """
    Scenario 1:
        Fraud Condition: Transactions with amounts greater than 220.
        Implementation: The function sets the TX_FRAUD column to 1 and the TX_FRAUD_SCENARIO column to 1 for transactions that meet the fraud condition.
    """
    transactions_data.loc[transactions_data['TX_AMOUNT'] > 220, 'TX_FRAUD'] = 1
    transactions_data.loc[transactions_data['TX_AMOUNT'] > 220, 'TX_FRAUD_SCENARIO'] = 1
    nb_frauds_scenario_1 = transactions_data['TX_FRAUD'].sum()
    print("Number of frauds from scenario 1: " + str(nb_frauds_scenario_1))

    # Scenario 2
    """Scenario 2: Fraud Condition: Compromised terminals. Implementation: For each day, two terminals are randomly 
    selected from the terminal_profiles_table. The function identifies transactions that occurred within a 28-day 
    window, involving the compromised terminals, and sets the TX_FRAUD column to 1 and the TX_FRAUD_SCENARIO column 
    to 2 for those transactions.
    """
    for day in range(transactions_data['TX_TIME_DAYS'].max()):
        compromised_terminals = terminals_data['TERMINAL_ID'].sample(n=2, random_state=day)

        compromised_transactions = transactions_data[(transactions_data['TX_TIME_DAYS'] >= day) &
                                                     (transactions_data['TX_TIME_DAYS'] < day + 28) &
                                                     (transactions_data['TERMINAL_ID'].isin(compromised_terminals))]

        transactions_data.loc[compromised_transactions.index, 'TX_FRAUD'] = 1
        transactions_data.loc[compromised_transactions.index, 'TX_FRAUD_SCENARIO'] = 2

    nb_frauds_scenario_2 = transactions_data['TX_FRAUD'].sum() - nb_frauds_scenario_1
    print("Number of frauds from scenario 2: " + str(nb_frauds_scenario_2))

    # Scenario 3
    """Scenario 3: Fraud Condition: Compromised customers. Implementation: For each day, three customers are randomly 
    selected from the customer_profiles_table. The function identifies transactions that occurred within a 14-day 
    window, involving the compromised customers. A random subset of one-third of the compromised transactions is 
    selected, and their TX_AMOUNT values are multiplied by 5. The TX_FRAUD column is set to 1, 
    and the TX_FRAUD_SCENARIO column is set to 3 for the selected transactions.
    """
    for day in range(transactions_data['TX_TIME_DAYS'].max()):
        compromised_customers = customer_profiles_data['CUSTOMER_ID'].sample(n=3, random_state=day).values

        compromised_transactions = transactions_data[(transactions_data['TX_TIME_DAYS'] >= day) &
                                                     (transactions_data['TX_TIME_DAYS'] < day + 14) &
                                                     (transactions_data['CUSTOMER_ID'].isin(compromised_customers))]

        nb_compromised_transactions = len(compromised_transactions)

        random.seed(day)
        index_frauds = random.sample(list(compromised_transactions.index.values),
                                     k=int(nb_compromised_transactions / 3))

        transactions_data.loc[index_frauds, 'TX_AMOUNT'] = transactions_data.loc[index_frauds, 'TX_AMOUNT'] * 5
        transactions_data.loc[index_frauds, 'TX_FRAUD'] = 1
        transactions_data.loc[index_frauds, 'TX_FRAUD_SCENARIO'] = 3

    nb_frauds_scenario_3 = transactions_data['TX_FRAUD'].sum() - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print("Number of frauds from scenario 3: " + str(nb_frauds_scenario_3))

    return transactions_data


def generate_dataset(n_customers: int, n_terminals: int, radius: float, start_date: str, nb_days: int) -> pd.DataFrame:
    """Generate a dataset with transactions data.
        params:
            n_customers: number of customers
            n_terminals: number of terminals
            radius: radius of the circle around the customer
            start_date: start date of the transactions
            nb_days: number of days to generate transactions for
            random_state: random state for reproducibility

        returns:
            transactions_data: a dataframe with transactions data
    """
    # Generate customer profiles
    customer_profiles_data = generate_customer_profiles_data(n_customers, random_state=0)

    # Generate terminal profiles
    terminals_data = generate_terminals_data(n_terminals, random_state=1)

    # Associate terminals to customers
    customers_terminals_data = map_terminals_to_customers(customer_profiles_data, terminals_data, radius)

    # Generate transactions data
    transactions_data = generate_transactions_data(customers_terminals_data, start_date, nb_days)

    # Generate frauds data
    fraud_transactions_df = generate_fraud_Scenarios_data(customers_terminals_data, terminals_data, transactions_data)

    return fraud_transactions_df
