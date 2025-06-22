from kedro.pipeline import node, Pipeline, pipeline
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs=["mushroom_data"],
            outputs="preprocessed_data",
            name="preprocess_data_node",
        ),
    ])
