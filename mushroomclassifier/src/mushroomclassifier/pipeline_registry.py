from kedro.pipeline import Pipeline
from .pipelines import process_data, train_model  # upewnij się, że to odpowiada nazwom folderów z pipeline'ami

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "preprocessing": process_data.create_pipeline(),
        "training": train_model.create_pipeline(),
        "__default__": train_model.create_pipeline() + train_model.create_pipeline(),
    }
