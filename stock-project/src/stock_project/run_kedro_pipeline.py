"""
This script sets up the project environment by adjusting the working directory
to the project root and updating the Python path to include the `src` directory.
It then configures the Kedro project and runs the specified pipeline.

If no pipeline_name is provided, the default pipeline "__default__" is run.
"""

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(project_root)

src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)


def run_pipeline(pipeline_name: str = "__default__"):
    """
    Configure and run a Kedro pipeline by name.

    Args:
        pipeline_name (str): Name of the pipeline to run. Defaults to "__default__".

    This function sets up the Kedro project environment and executes the
    specified pipeline within a Kedro session.
    """
    configure_project("stock_project")
    with KedroSession.create(project_path=project_root) as session:
        session.run(pipeline_name=pipeline_name)