from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from zenml import step


@step(experiment_tracker="mlflow_tracker")
def model_evaluator(
    trained_base_models: Dict[str, Any],
    meta_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate the ensemble model and individual base models on the original scale and log to MLflow.

    Args:
        trained_base_models: Dictionary of trained base models.
        meta_model: The trained meta-model.
        X_test: Held-out preprocessed test features.
        y_test: Held-out test targets (in log-scale).

    Returns:
        A dictionary containing RMSE, MAE, and R2 metrics on the original scale.
    """
    print("Starting model evaluation on original scale with MLflow logging...")

    if X_test.empty:
        print("Test data is empty, skipping evaluation.")
        return {}

    # Convert y_test back to original scale (Lakhs)
    y_test_orig = np.expm1(y_test)
    metrics = {}

    for name, model in trained_base_models.items():
        y_pred_log = model.predict(X_test)
        y_pred_orig = np.expm1(y_pred_log)
        
        rmse = mean_squared_error(y_test_orig, y_pred_orig)**0.5
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_r2"] = r2
        
        # Log to MLflow
        mlflow.log_metric(f"{name}_rmse", rmse)
        mlflow.log_metric(f"{name}_mae", mae)
        mlflow.log_metric(f"{name}_r2", r2)
        
        print(
            f"Base Model {name} - RMSE: {rmse:.4f}, "
            f"MAE: {mae:.4f}, R2: {r2:.4f}"
        )

    meta_features_test = pd.DataFrame(index=X_test.index)
    for name, model in trained_base_models.items():
        meta_features_test[f"{name}_pred"] = model.predict(X_test)

    y_pred_ensemble_log = meta_model.predict(meta_features_test)
    y_pred_ensemble_orig = np.expm1(y_pred_ensemble_log)
    
    ensemble_rmse = mean_squared_error(y_test_orig, y_pred_ensemble_orig)**0.5
    ensemble_mae = mean_absolute_error(y_test_orig, y_pred_ensemble_orig)
    ensemble_r2 = r2_score(y_test_orig, y_pred_ensemble_orig)
    
    metrics["ensemble_rmse"] = ensemble_rmse
    metrics["ensemble_mae"] = ensemble_mae
    metrics["ensemble_r2"] = ensemble_r2
    
    # Log to MLflow
    mlflow.log_metric("ensemble_rmse", ensemble_rmse)
    mlflow.log_metric("ensemble_mae", ensemble_mae)
    mlflow.log_metric("ensemble_r2", ensemble_r2)
    
    print(
        f"Ensemble Model - RMSE: {ensemble_rmse:.4f}, "
        f"MAE: {ensemble_mae:.4f}, R2: {ensemble_r2:.4f}"
    )

    print("Model evaluation complete. Metrics logged to MLflow.")
    return metrics
