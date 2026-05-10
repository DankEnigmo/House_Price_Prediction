import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from src.mlflow_diagnostics import print_mlflow_context
from xgboost import XGBRegressor
from zenml import step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_PARAMS_PATH = PROJECT_ROOT / "tuning_results" / "best_params_99acres.json"


def _merge_params(defaults: Dict[str, Any], tuned: Dict[str, Any]) -> Dict[str, Any]:
    return {**defaults, **tuned}


@step(experiment_tracker="mlflow_tracker")
def model_training_and_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[
    Annotated[Dict[str, Any], "trained_base_models"],
    Annotated[Any, "meta_model"],
]:
    """
    Train base models and a meta-model using Out-of-Fold (OOF) stacking and log to MLflow.
    Loads tuned parameters if available.

    Args:
        X_train: Preprocessed training features.
        y_train: Training target values.

    Returns:
        A dictionary of trained base models and the trained meta-model.
    """
    print("Starting model training and stacking with OOF...")
    print_mlflow_context("model_training_and_stacking")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    # Default parameters
    tuned_params = {}
    if os.path.exists(BEST_PARAMS_PATH):
        print(f"Loading tuned parameters from {BEST_PARAMS_PATH}...")
        with open(BEST_PARAMS_PATH, "r") as f:
            tuned_params = json.load(f)

    xgboost_params = _merge_params(
        {"random_state": 42, "n_jobs": -1},
        tuned_params.get("xgboost", {}),
    )
    lightgbm_params = _merge_params(
        {"random_state": 42, "n_jobs": -1, "verbose": -1},
        tuned_params.get("lightgbm", {}),
    )
    catboost_params = _merge_params(
        {"random_seed": 42, "verbose": 0, "allow_writing_files": False},
        tuned_params.get("catboost", {}),
    )

    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(**xgboost_params),
        "LightGBM": LGBMRegressor(**lightgbm_params),
        "CatBoost": CatBoostRegressor(**catboost_params),
    }

    # Initialize KFold for OOF predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = pd.DataFrame(index=X_train.index)

    # Generate OOF predictions for each base model
    for name, model in base_models.items():
        print(f"Generating OOF predictions for: {name}")
        oof_preds = np.zeros(len(X_train))
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            # Clone and fit on training folds
            model_clone = clone(model)
            model_clone.fit(X_fold_train, y_fold_train)
            
            # Predict on validation fold
            oof_preds[val_idx] = model_clone.predict(X_fold_val)
            
        meta_features[f"{name}_pred"] = oof_preds

    # Train the meta-model on OOF predictions
    print("Training meta-model (RidgeCV) on OOF features...")
    meta_model = RidgeCV(alphas=np.logspace(-2, 2, 9))
    meta_model.fit(meta_features, y_train)
    
    # Log meta-model to MLflow
    mlflow.sklearn.log_model(meta_model, "meta_model")

    # Train final base models on the full training set and log them
    trained_base_models = {}
    for name, model in base_models.items():
        print(f"Training final base model on full data: {name}")
        model.fit(X_train, y_train)
        trained_base_models[name] = model
        
        # Explicit model logging by flavor
        if name == "RandomForest":
            mlflow.sklearn.log_model(model, f"{name.lower()}_model")
        elif name == "XGBoost":
            mlflow.xgboost.log_model(model, f"{name.lower()}_model")
        elif name == "LightGBM":
            mlflow.lightgbm.log_model(model, f"{name.lower()}_model")
        elif name == "CatBoost":
            mlflow.catboost.log_model(model, f"{name.lower()}_model")

    print("Model training and stacking complete. Models logged to MLflow.")
    return trained_base_models, meta_model
