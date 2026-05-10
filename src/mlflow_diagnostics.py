from __future__ import annotations

import mlflow


def print_mlflow_context(step_name: str) -> None:
    """Print the active MLflow context that ZenML configured for a tracked step."""
    tracking_uri = mlflow.get_tracking_uri()
    active_run = mlflow.active_run()

    print(f"MLflow context [{step_name}] tracking_uri: {tracking_uri}")
    if active_run is None:
        print(f"MLflow context [{step_name}] active_run: none")
        return

    experiment = mlflow.get_experiment(active_run.info.experiment_id)
    experiment_name = experiment.name if experiment is not None else "unknown"
    print(f"MLflow context [{step_name}] run_id: {active_run.info.run_id}")
    print(f"MLflow context [{step_name}] experiment: {experiment_name}")
    mlflow.set_tag("zenml_step", step_name)
