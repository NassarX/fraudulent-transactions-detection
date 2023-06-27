"""Project pipelines."""
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from typing import Dict
from fraud_detection.pipelines.etl_app.pipeline import create_etl_pipeline
from fraud_detection.pipelines.ml_app.pipeline import create_ml_pipeline
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    # ETL Pipeline
    etl_pipeline = create_etl_pipeline()
    etl_pipelines = etl_pipeline.only_nodes_with_tags("etl")
    etl_generation_pipeline = etl_pipeline.only_nodes_with_tags("etl_generate")
    etl_transformation_pipeline = etl_pipeline.only_nodes_with_tags("etl_transform")
    etl_preprocessing_pipeline = etl_pipeline.only_nodes_with_tags("etl_preprocess")
    etl_exploration_pipeline = etl_pipeline.only_nodes_with_tags("etl_explore")

    # Machine Learning Pipeline
    ml_pipeline = create_ml_pipeline()
    ml_pipelines = ml_pipeline.only_nodes_with_tags("ml")
    ml_training_pipeline = ml_pipeline.only_nodes_with_tags("ml_train", "ml_predict")
    ml_inference_pipeline = ml_pipeline.only_nodes_with_tags("ml_eval", "ml_inference")

    return {
        "__default__": etl_pipeline + ml_pipeline,
        "etl": etl_pipelines,
        "ml": ml_pipelines,
        "etl_generation": etl_generation_pipeline,
        "etl_transformation": etl_transformation_pipeline,
        "etl_preprocessing": etl_preprocessing_pipeline,
        "etl_exploration": etl_exploration_pipeline,
        "ml_training": ml_training_pipeline,
        "ml_inference": ml_inference_pipeline,
    }
