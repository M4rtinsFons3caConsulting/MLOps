"""
This script defines and registers Prefect deployments for MLOps workflows.

It builds deployments from flows defined in `flows.kedro_flows`, each with its own schedule and work queue.
The deployments cover unit tests, production training, model assessment, daily prediction, and drift analysis.

Deployments created:
- Unit tests: runs nightly at 22:00 (Europe/Lisbon)
- Production training: runs daily at 07:00
- Quarterly model assessment: runs on the 1st of Jan, Apr, Jul, and Oct at 00:00
- Production prediction: runs daily at 08:00
- Drift analysis: runs monthly on the 1st at 17:00

After running this script, deployments are applied and will appear in the Prefect UI.
"""

from prefect.deployments import Deployment
from prefect.client.schemas.schedules import CronSchedule

from flows.kedro_flows import (
    flow_data_unit_tests,
    flow_production_training,
    flow_production_assessment,
    flow_production_prediction,
    flow_drift_analysis,
)

# Unit Tests – Nightly at 10 PM
Deployment.build_from_flow(
    flow=flow_data_unit_tests,
    name="unit-test-nightly",
    work_queue_name="default",
    schedule=CronSchedule(
        cron="0 22 * * *", 
        timezone="Europe/Lisbon"
    )
).apply()

# Production Training – Daily at 7 AM
Deployment.build_from_flow(
    flow=flow_production_training,
    name="full-processing-morning",
    work_queue_name="default",
    schedule=CronSchedule(
        cron="0 7 * * *", 
        timezone="Europe/Lisbon"
    )
).apply()

# Quarterly Model Assessment – 1st Jan, Apr, Jul, Oct at midnight
Deployment.build_from_flow(
    flow=flow_production_assessment,
    name="model-assessment-quarterly",
    work_queue_name="default",
    schedule=CronSchedule(
        cron="0 0 1 1,4,7,10 *", 
        timezone="Europe/Lisbon"
    )
).apply()

# Production Prediction – Daily at 8 AM
Deployment.build_from_flow(
    flow=flow_production_prediction,
    name="production-prediction-daily",
    work_queue_name="default",
    schedule=CronSchedule(
        cron="0 8 * * *",
        timezone="Europe/Lisbon"
    )
).apply()

# Drift Analysis – Monthly at 5 PM
Deployment.build_from_flow(
    flow=flow_drift_analysis,
    name="drift-check-monthly",
    work_queue_name="default",
    schedule=CronSchedule(
        cron="0 17 1 * *",
        timezone="Europe/Lisbon"
    )
).apply()