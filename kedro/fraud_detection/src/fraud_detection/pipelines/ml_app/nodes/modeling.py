import time

import pandas as pd
from kedro.config import ConfigLoader
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from typing import Any, Tuple, List, Dict, Callable
from sklearn import tree, ensemble, svm, linear_model
from xgboost import XGBClassifier


def fit_predict_models(
    input_features: list,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    output_feature="TX_FRAUD",
) -> tuple[Any, Any]:
    """Fits and predicts multiple models on the training and test datasets
    :param input_features: List of input features
    :param x_train: DataFrame containing training data
    :param x_test: DataFrame containing test data
    :param y_train: DataFrame containing training labels
    :param output_feature: Name of output feature
    :return: Tuple containing trained models and predictions
    """
    conf_loader = ConfigLoader("conf")
    parameters = conf_loader.get("parameters.yml", "parameters")
    models = parameters["models"]

    # Initialize dictionary to store trained models
    trained_models = {}
    predictions = {}
    for model_name in models:
        # Get model configs
        configs = extract_model_configs(model_name, parameters)

        # build classifier
        classifier = build_classifier(model_name, configs)

        # scales input data for logistic regression and svm
        if model_name in ["logistic_regression", "svm"]:
            (x_train, x_test) = scaleData(x_train, x_test, input_features)

        # Fit model
        fitted_model, training_time = fit_model(
            model=classifier,
            input_features=input_features,
            x_train=x_train,
            y_train=y_train,
            output_feature=output_feature,
        )

        # Store model
        trained_models[f"{model_name}.pkl"] = fitted_model

        # Predict
        test_predictions = predict(fitted_model, x_test, input_features)

        # Store predictions
        predictions[f"{model_name}_predictions.csv"] = pd.DataFrame(test_predictions)

    return trained_models, predictions


def fit_model(
    model,
    input_features: list,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    output_feature,
):
    """Fits a model on the training dataset and returns the trained model and training time
    :param model: Model to fit
    :param input_features: List of input features
    :param x_train: DataFrame containing training data
    :param y_train: DataFrame containing training labels
    :param output_feature: Name of output feature
    :return: Tuple containing trained model and training time
    """
    # We first train the classifier using the `fit` method, and pass as arguments the input and output features
    start_time = time.time()
    model.fit(x_train[input_features], y_train[output_feature])
    training_time = time.time() - start_time  ## INFO logging

    return model, training_time


def predict(model, x_test: pd.DataFrame, input_features: list) -> Dict[str, Any]:
    """Predicts using a trained model on the test dataset
    :param model: Trained model
    :param x_test: DataFrame containing test data
    :param input_features: List of input features
    :return: Dictionary containing model predictions
    """
    start_time = time.time()
    predictions = model.predict_proba(x_test[input_features])[:, 1]
    prediction_time = time.time() - start_time

    return predictions


def extract_model_configs(model: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts model configs from parameters.yml
    :param model: Name of model
    :param parameters: Dictionary containing parameters
    :return: Dictionary containing model configs
    """
    configs = {}
    model_configs = parameters[f"{model}_configs"]
    if model_configs is not None:
        configs = {k: v for d in model_configs for k, v in d.items()}
    return configs


def scaleData(
    train_df: pd.DataFrame, test_df: pd.DataFrame, input_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scales the input features in both train and test datasets
    :param train_df: DataFrame containing training data
    :param test_df: DataFrame containing test data
    :param input_features: List of input features
    :return: Tuple containing scaled train and test datasets
    """
    # Create a scaler object
    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(train_df[input_features])

    # Scale the input features in both train and test datasets
    train_df_scaled = pd.DataFrame(
        scaler.transform(train_df[input_features]), columns=input_features
    )
    test_df_scaled = pd.DataFrame(
        scaler.transform(test_df[input_features]), columns=input_features
    )

    # Combine the scaled features with the remaining columns
    train_df_scaled = pd.concat(
        [train_df_scaled, train_df.drop(columns=input_features)], axis=1
    )
    test_df_scaled = pd.concat(
        [test_df_scaled, test_df.drop(columns=input_features)], axis=1
    )

    return train_df_scaled, test_df_scaled


def build_classifier(
    classifier_name: str,
    configs: Dict[str, Any],
):
    """Builds a classifier based on the classifier name
    :param classifier_name:  of classifier
    :param configs: Dictionary containing classifier configs
    :return: Classifier
    """

    classifier = tree.DecisionTreeClassifier()

    if classifier_name == "logistic_regression":
        classifier = linear_model.LogisticRegression(**configs)
    if classifier_name == "decision_tree_depth_2":
        classifier = tree.DecisionTreeClassifier(**configs)
    if classifier_name == "decision_tree_depth_unlimited":
        classifier = tree.DecisionTreeClassifier(**configs)
    elif classifier_name == "random_forest":
        classifier = ensemble.RandomForestClassifier(**configs)
    elif classifier_name == "xgboost":
        classifier = XGBClassifier(**configs)

    return classifier
