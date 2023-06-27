from kedro.pipeline import Pipeline, node

from fraud_detection.pipelines.etl_app.nodes.data_generation import (
    generate_customer_profiles_data,
    generate_terminals_data,
    map_terminals_to_customers,
    generate_transactions_data,
    generate_fraud_Scenarios_data,
)

from fraud_detection.pipelines.etl_app.nodes.feature_transformation import (
    transform_datetime_features,
    transform_customer_features,
    transform_terminal_features,
)

from fraud_detection.pipelines.etl_app.nodes.data_preprocessing import (
    split_transaction_data_into_batches,
    process_daily_transaction_data,
    split_train_test_data,
)

from fraud_detection.pipelines.etl_app.nodes.data_exploration import (
    plot_fraud_distribution,
    plot_transactions_distribution,
    plot_transactions_daily_stats,
)


def create_etl_pipeline(**kwargs):
    # Data Generations Nodes
    pipeline_data_generation = Pipeline(
        [
            node(
                func=generate_customer_profiles_data,
                inputs=dict(n_customers="params:n_customers"),
                outputs="customers_data",
                tags=["etl", "etl_generate"],
                name="node_generate_customer_profiles_data",
            ),
            node(
                func=generate_terminals_data,
                inputs=dict(n_terminals="params:n_customers"),
                outputs="terminals_data",
                tags=["etl", "etl_generate"],
                name="node_generate_terminals_data",
            ),
            node(
                func=map_terminals_to_customers,
                inputs=dict(
                    customers_data="customers_data",
                    terminals_data="terminals_data",
                    radius="params:radius",
                ),
                outputs="customers_terminals_data",
                tags=["etl", "etl_generate"],
                name="node_map_terminals_to_customers",
            ),
            node(
                func=generate_transactions_data,
                inputs=dict(
                    customers_terminals_data="customers_terminals_data",
                    start_date="params:start_date",
                    nb_days="params:nb_days",
                ),
                outputs="transactions_data",
                tags=["etl", "etl_generate"],
                name="node_generate_transactions_data",
            ),
            node(
                func=generate_fraud_Scenarios_data,
                inputs=dict(
                    customer_profiles_data="customers_terminals_data",
                    terminals_data="terminals_data",
                    transactions_data="transactions_data",
                ),
                outputs="fraud_transactions_data",
                tags=["etl", "etl_generate"],
                name="node_generate_fraud_Scenarios_data",
            ),
        ]
    )

    # Features Transformations Nodes
    pipeline_feature_transformation = Pipeline(
        [
            node(
                func=transform_datetime_features,
                inputs=dict(transactions_data="fraud_transactions_data"),
                outputs="fraud_transactions_transform_v1_data",
                tags=["etl", "etl_transform"],
                name="node_transform_datetime_features",
            ),
            node(
                func=transform_customer_features,
                inputs=dict(transactions_data="fraud_transactions_transform_v1_data"),
                outputs="fraud_transactions_transform_v2_data",
                tags=["etl", "etl_transform"],
                name="node_transform_customer_features",
            ),
            node(
                func=transform_terminal_features,
                inputs=dict(transactions_data="fraud_transactions_transform_v2_data"),
                outputs="fraud_transactions_transform_v3_data",
                tags=["etl", "etl_transform"],
                name="node_transform_terminal_features",
            ),
        ]
    )

    # Data Preprocessing Nodes
    pipeline_data_preprocessing = Pipeline(
        [
            node(
                func=split_transaction_data_into_batches,
                inputs=dict(
                    transactions_data="fraud_transactions_transform_v3_data",
                    start_date="params:start_date",
                ),
                outputs="daily_transactions_data",
                tags=["etl", "etl_preprocess"],
                name="node_split_transaction_data_into_batches",
            ),
            node(
                func=process_daily_transaction_data,
                inputs=dict(partitioned_input="daily_transactions_data"),
                outputs="simulated_transactions_data",
                tags=["etl", "etl_preprocess"],
                name="node_merge_daily_transactions_data",
            ),
            node(
                func=split_train_test_data,
                inputs=dict(
                    processed_df="simulated_transactions_data",
                    start_date_train="params:start_train_date",
                    delta_train="params:delta_train",
                    delta_delay="params:delta_delay",
                    delta_test="params:delta_test",
                ),
                outputs=["x_train", "y_train", "x_test", "y_test"],
                tags=["etl", "etl_preprocess"],
                name="node_split_train_test_data",
            ),
        ]
    )

    # Data Exploration Nodes
    pipeline_exploration = Pipeline(
        [
            node(
                func=plot_fraud_distribution,
                inputs=dict(transactions="simulated_transactions_data"),
                outputs="fraud_distribution_plot",
                tags=["etl", "etl_explore"],
            ),
            node(
                func=plot_transactions_distribution,
                inputs=dict(transactions="simulated_transactions_data"),
                outputs="transactions_distributions_plot",
                tags=["etl", "etl_explore"],
            ),
            node(
                func=plot_transactions_daily_stats,
                inputs=dict(transactions="simulated_transactions_data"),
                outputs="transactions_daily_stats_plot",
                tags=["etl", "etl_explore"],
            ),
        ]
    )

    return (
        pipeline_data_generation
        + pipeline_feature_transformation
        + pipeline_data_preprocessing
        + pipeline_exploration
    )
