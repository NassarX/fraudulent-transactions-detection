import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_fraud_distribution(transactions: pd.DataFrame) -> None:
    """
    Plots the distribution of fraud and non-fraud transactions
    :param transactions: transactions dataframe
    :return: None
    """
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Count the occurrences of fraud and non-fraud transactions
    fraud_counts = transactions["TX_FRAUD"].value_counts()

    # Plot the count plot
    sns.countplot(x="TX_FRAUD", data=transactions, ax=ax)

    # Set the labels and title
    ax.set_xlabel("Transaction Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticklabels(["Non-Fraud", "Fraud"], rotation=0)
    ax.set_title("Distribution of Fraud and Non-Fraud Transactions", fontsize=14)

    # Annotate the bars with the count values
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return fig


def plot_transactions_distribution(transactions: pd.DataFrame):
    """
    Plots the distribution of transaction amounts & times
    :param transactions: transactions dataframe
    :return: None
    """
    distribution_amount_times_fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    amount_val = transactions[transactions.TX_TIME_DAYS < 10]["TX_AMOUNT"].values
    time_val = transactions[transactions.TX_TIME_DAYS < 10]["TX_TIME_SECONDS"].values

    sample_size = min(
        10000, len(amount_val)
    )  # Limit the sample size to the population size

    amount_sample = np.random.choice(amount_val, size=sample_size, replace=False)
    time_sample = np.random.choice(time_val, size=sample_size, replace=False)

    # Plot distribution of transaction amounts
    sns.distplot(amount_sample, ax=ax[0], color="r", hist=True, kde=False)
    ax[0].set_title("Distribution of transaction amounts", fontsize=14)
    ax[0].set_xlim([min(amount_sample), max(amount_sample)])
    ax[0].set(xlabel="Amount", ylabel="Number of transactions")

    # We divide the time variables by 86400 to transform seconds to days in the plot
    sns.distplot(
        time_sample / 86400, ax=ax[1], color="b", bins=100, hist=True, kde=False
    )
    ax[1].set_title("Distribution of transaction times", fontsize=14)
    ax[1].set_xlim([min(time_sample / 86400), max(time_sample / 86400)])
    ax[1].set_xticks(range(10))
    ax[1].set(xlabel="Time (days)", ylabel="Number of transactions")

    return distribution_amount_times_fig


def plot_transactions_daily_stats(transactions: pd.DataFrame):
    """
    Plots the daily statistics of transaction amounts & times
    :param transactions: transactions dataframe
    :return: None
    """
    # Get the daily statistics
    nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day = get_stats(transactions)
    n_days = len(nb_tx_per_day)
    tx_stats = pd.DataFrame(
        {"value": pd.concat([nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day])}
    )
    tx_stats["stat_type"] = (
        ["nb_tx_per_day"] * n_days
        + ["nb_fraud_per_day"] * n_days
        + ["nb_fraudcard_per_day"] * n_days
    )
    tx_stats = tx_stats.reset_index()

    sns.set(style="darkgrid")
    sns.set(font_scale=1.4)
    fraud_and_transactions_stats_fig = plt.gcf()

    fraud_and_transactions_stats_fig.set_size_inches(20, 8)
    sns_plot = sns.lineplot(
        x="TX_TIME_DAYS",
        y="value",
        data=tx_stats,
        hue="stat_type",
        hue_order=["nb_tx_per_day", "nb_fraud_per_day", "nb_fraudcard_per_day"],
        legend=False,
    )

    sns_plot.set_title(
        "Total transactions, and number of fraudulent transactions \n and number of compromised cards per day",
        fontsize=20,
    )
    sns_plot.set(
        xlabel="Number of days since beginning of data generation", ylabel="Number"
    )
    sns_plot.set_ylim([0, 2000])

    labels_legend = [
        "# transactions per day",
        "# fraudulent txs per day",
        "# fraudulent cards per day",
    ]
    sns_plot.legend(loc="upper left", labels=labels_legend, ncol=3, fontsize=15)

    return fraud_and_transactions_stats_fig


def get_stats(transactions_df):
    # Number of transactions per day
    nb_tx_per_day = transactions_df.groupby(["TX_TIME_DAYS"])["CUSTOMER_ID"].count()

    # Number of fraudulent transactions per day
    nb_fraud_per_day = transactions_df.groupby(["TX_TIME_DAYS"])["TX_FRAUD"].sum()

    # Number of fraudulent cards per day
    nb_fraudcard_per_day = (
        transactions_df[transactions_df["TX_FRAUD"] > 0]
        .groupby(["TX_TIME_DAYS"])
        .CUSTOMER_ID.nunique()
    )

    return nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day
