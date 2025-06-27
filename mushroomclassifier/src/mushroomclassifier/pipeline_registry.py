from kedro.pipeline import Pipeline
from mushroomclassifier.pipelines.process_data import create_pipeline as process_data_pipeline
from mushroomclassifier.pipelines.train_model import create_pipeline as train_model_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "process_data": process_data_pipeline(),
        "train_model": train_model_pipeline(),
        "__default__": process_data_pipeline() + train_model_pipeline()
    }

