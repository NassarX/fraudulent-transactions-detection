from kedro.pipeline import Pipeline, node
from fraud_detection.pipelines.ml_app.nodes.modeling import (
    fit_predict_models,
)

from fraud_detection.pipelines.ml_app.nodes.model_eval import (
    evaluate_models,
)
def create_ml_pipeline(**kwargs):

    pipeline_modeling = Pipeline(
        [
            node(
                func=fit_predict_models,
                inputs=dict(
                    x_train="x_train",
                    x_test="x_test",
                    y_train="y_train",
                    input_features="params:input_features",
                    output_feature="params:output_feature",
                ),
                outputs=["trained_models", "models_predictions"],
                tags=["ml_train", "ml_predict","ml"],
            ),
        ]
    )

    pipeline_inference = Pipeline(
        [
            node(
                func=evaluate_models,
                inputs=dict(predictions="models_predictions", y_test="y_test"),
                outputs="models_metrics",
                tags=["ml_inference", "ml_eval", "ml"]
            ),
        ]
    )

    return (
        pipeline_modeling
        + pipeline_inference
    )
