from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def run_pipeline(pipeline_name: str = "__default__"):
    configure_project("bank_full_project")
    with KedroSession.create(project_path=".") as session:
        session.run(pipeline_name=pipeline_name)