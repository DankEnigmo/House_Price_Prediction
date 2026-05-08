from zenml import pipeline

from steps.data_ingestion_step import data_ingestion
from steps.data_validation_step import data_validation
from steps.feature_engineering_step import feature_engineering
from steps.model_evaluator_step import model_evaluator
from steps.model_training_and_stacking_step import model_training_and_stacking


@pipeline
def house_price_prediction_pipeline():
    """
    ZenML pipeline for Indian House Price Prediction with a hybrid ensemble model.
    """
    raw_data = data_ingestion()
    validated_data = data_validation(df=raw_data)
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df=validated_data)

    trained_base_models, meta_model = model_training_and_stacking(
        X_train=X_train,
        y_train=y_train,
    )

    metrics = model_evaluator(
        trained_base_models=trained_base_models,
        meta_model=meta_model,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Pipeline finished with metrics: {metrics}")


if __name__ == "__main__":
    house_price_prediction_pipeline.with_options(enable_cache=False)()
