"""Project pipelines."""
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from kedro.pipeline import Pipeline
from platform import python_version
from typing import Dict
from fraud_detection.pipelines.etl_app.pipeline import create_etl_pipeline
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    etl_pipeline = create_etl_pipeline()
    etl_generation_pipeline = etl_pipeline.only_nodes_with_tags("etl_generate")
    etl_transformation_pipeline = etl_pipeline.only_nodes_with_tags("etl_transform")
    etl_preprocessing_pipeline = etl_pipeline.only_nodes_with_tags("etl_preprocess")

    return {
        "__default__": etl_pipeline,  # Set the default pipeline to the complete etl_pipeline
        "etl_generation": etl_generation_pipeline,
        "etl_transformation": etl_transformation_pipeline,
        "etl_preprocessing": etl_preprocessing_pipeline,
    }
