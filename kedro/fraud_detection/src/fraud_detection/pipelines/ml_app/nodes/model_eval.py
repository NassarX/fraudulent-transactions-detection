import time

import pandas as pd
from typing import Any, Tuple, List, Dict, Callable
from sklearn import metrics


def evaluate_models(predictions: Dict[str, Callable[[], Any]], y_test: pd.DataFrame) -> dict[
    str, dict[str, Any]]:
    """
    Evaluates model performance by comparing predictions to actual values
    :param predictions: dictionary of predictions
    :param y_test: actual values
    :return: dictionary of model performance metrics
    """

    # log model performance metrics
    models_performance_metrics = {}
    for prediction, prediction_load_func in sorted(predictions.items()):
        if prediction == ".gitkeep":
            continue

        # load prediction data
        prediction_data = prediction_load_func()

        # get model name
        model_name = prediction.split("_predictions")[0]

        # get model performance metrics
        AUC_ROC = metrics.roc_auc_score(y_test, prediction_data)
        AUC_PR = metrics.average_precision_score(y_test, prediction_data)
        log_loss = metrics.log_loss(y_test, prediction_data)

        # log model performance metrics
        model_metrics = {
            f"{model_name}_AUC_ROC": {"value": AUC_ROC, "timestamp": time.time(), "step": 1},
            f"{model_name}_AUC_PR": {"value": AUC_PR, "timestamp": time.time(), "step": 1},
            f"{model_name}_log_loss": {"value": log_loss, "timestamp": time.time(), "step": 1}
        }

        models_performance_metrics.update(model_metrics)

    return models_performance_metrics