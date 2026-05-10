import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from src.mlflow_diagnostics import print_mlflow_context
from xgboost import XGBRegressor
from zenml import step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_RESULTS_DIR = PROJECT_ROOT / "tuning_results"
BEST_PARAMS_PATH = TUNING_RESULTS_DIR / "best_params_99acres.json"
DB_PATH = f"sqlite:///{TUNING_RESULTS_DIR}/optuna_99acres.db"
MAX_TUNING_ROWS = 50_000
N_TRIALS = 50


@step(experiment_tracker="mlflow_tracker")
def hp_tuning_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[Dict[str, Any], "best_parameters"]:
    """
    Hyperparameter tuning step using Optuna with Successive Halving and Adaptive Sampling.
    """
    os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
    print_mlflow_context("hp_tuning_step")

    if len(X_train) > MAX_TUNING_ROWS:
        X_tune = X_train.sample(MAX_TUNING_ROWS, random_state=42)
        y_tune = y_train.loc[X_tune.index]
    else:
        X_tune = X_train
        y_tune = y_train

    print(f"Optuna tuning rows: {len(X_tune)}")
    print(f"Optuna trials per model: {N_TRIALS}")

    def objective(trial, model_name):
        if model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "early_stopping_rounds": 50,
            }
            model_class = XGBRegressor
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
            }
            model_class = LGBMRegressor
        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_seed": 42,
                "verbose": 0,
                "allow_writing_files": False,
            }
            model_class = CatBoostRegressor

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tune)):
            X_fold_train, X_fold_val = X_tune.iloc[train_idx], X_tune.iloc[val_idx]
            y_fold_train, y_fold_val = y_tune.iloc[train_idx], y_tune.iloc[val_idx]

            model = model_class(**params)

            fit_params = {"eval_set": [(X_fold_val, y_fold_val)]}
            if model_name == "xgboost":
                fit_params["verbose"] = False
            elif model_name == "catboost":
                fit_params["verbose"] = 0

            model.fit(X_fold_train, y_fold_train, **fit_params)

            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(np.mean((y_fold_val - y_pred) ** 2))
            cv_scores.append(rmse)

            # Report to Optuna for pruning
            trial.report(rmse, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    best_params = {}
    for model_name in ["xgboost", "lightgbm", "catboost"]:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(
            study_name=f"{model_name}_tuning",
            storage=DB_PATH,
            direction="minimize",
            sampler=TPESampler(multivariate=True, seed=42),
            pruner=SuccessiveHalvingPruner(),
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(trial, model_name), 
            n_trials=N_TRIALS, 
            n_jobs=4
        )
        best_params[model_name] = study.best_params
        print(f"Best params for {model_name}: {study.best_params}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params
