from prefect.client.schemas.schedules import CronSchedule
from flows.kedro_flows import (
    flow_data_unit_tests,
    flow_production_training,
    flow_production_assessment,
    flow_production_prediction,
    flow_drift_analysis,
)

# Unit Tests – Nightly at 10 PM
deployment1 = flow_data_unit_tests.deploy(
    name="unit-test-nightly",
    work_pool_name="docker-pool",
    schedule=CronSchedule(
        cron="0 22 * * *",
        timezone="Europe/Lisbon"
    ),
    image="example:latest"
)

# # Production Training – Daily at 7 AM
# deployment2 = flow_production_training.deploy(
#     name="full-processing-morning",
#     work_pool_name="docker-pool",
#     schedule=CronSchedule(
#         cron="0 7 * * *",
#         timezone="Europe/Lisbon"
#     ),
#     image="example:latest"
# )

# # Quarterly Model Assessment – 1st Jan, Apr, Jul, Oct at midnight
# deployment3 = flow_production_assessment.deploy(
#     name="model-assessment-quarterly",
#     work_pool_name="docker-pool",
#     schedule=CronSchedule(
#         cron="0 0 1 1,4,7,10 *",
#         timezone="Europe/Lisbon"
#     ),
#     image="example:latest"
# )

# # Production Prediction – Daily at 8 AM
# deployment4 = flow_production_prediction.deploy(
#     name="production-prediction-daily",
#     work_pool_name="docker-pool",
#     schedule=CronSchedule(
#         cron="0 8 * * *",
#         timezone="Europe/Lisbon"
#     ),
#     image="example:latest"
# )

# # Drift Analysis – Monthly at 5 PM
# deployment5 = flow_drift_analysis.deploy(
#     name="drift-check-monthly",
#     work_pool_name="docker-pool",
#     schedule=CronSchedule(
#         cron="0 17 1 * *",
#         timezone="Europe/Lisbon"
#     ),
#     image="example:latest"
# )