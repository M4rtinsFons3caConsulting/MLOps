"""
This is a pipeline 'data_wrangling'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import combine_files

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=combine_files
            ,inputs="params:raw_data_dir"
            ,outputs="ingested_data"
            ,name="combine_raw_files_node"
        )
    ])
