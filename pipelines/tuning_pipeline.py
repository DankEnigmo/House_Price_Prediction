from zenml import pipeline
from steps.data_ingestion_step import data_ingestion
from steps.data_validation_step import data_validation
from steps.feature_engineering_step import feature_engineering
from steps.hp_tuning_step import hp_tuning_step

@pipeline
def house_price_tuning_pipeline():
    """
    Pipeline for hyperparameter tuning using Optuna.
    """
    raw_data = data_ingestion()
    validated_data = data_validation(df=raw_data)
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df=validated_data)
    best_params = hp_tuning_step(X_train=X_train, y_train=y_train)

if __name__ == "__main__":
    house_price_tuning_pipeline.with_options(enable_cache=False)()
