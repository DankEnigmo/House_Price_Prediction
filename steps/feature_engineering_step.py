from typing import Annotated, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import step


TARGET_COLUMN = "Price_in_Lakhs"
LEAKAGE_COLUMNS = ["ID", "Price_per_SqFt"]
AMENITIES_COLUMN = "Amenities"
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIGH_CARDINALITY_THRESHOLD = 50
TARGET_ENCODING_SMOOTHING = 10


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class HousingFeaturePreprocessor:
    """Train-fitted preprocessor for the housing dataset."""

    def __init__(self) -> None:
        self.amenity_values_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.high_cardinality_columns_: list[str] = []
        self.target_maps_: dict[str, dict[str, float]] = {}
        self.global_mean_: float = 0.0
        self.column_transformer_: ColumnTransformer | None = None

    def fit(self, data: pd.DataFrame) -> "HousingFeaturePreprocessor":
        # Note: data must contain the TARGET_COLUMN (ideally in log-scale)
        self.global_mean_ = data[TARGET_COLUMN].mean()
        
        features = self._prepare_features(data, fit=True)
        categorical_candidates = features.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
        
        self.high_cardinality_columns_ = [
            column
            for column in categorical_candidates
            if features[column].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
        ]
        
        # Calculate Smoothed Target Encoding maps
        self.target_maps_ = {}
        for column in self.high_cardinality_columns_:
            # Group by category and calculate count and mean of target
            stats = data.groupby(column)[TARGET_COLUMN].agg(["count", "mean"])
            
            # Apply smoothing formula
            # smoothed = (count * mean + m * global_mean) / (count + m)
            smoothed = (
                stats["count"] * stats["mean"] + TARGET_ENCODING_SMOOTHING * self.global_mean_
            ) / (stats["count"] + TARGET_ENCODING_SMOOTHING)
            
            self.target_maps_[column] = smoothed.to_dict()
            
        features = self._apply_target_encoding(features)

        self.numeric_columns_ = features.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_columns_ = [
            column for column in features.columns if column not in self.numeric_columns_
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_columns_),
                ("categorical", categorical_pipeline, self.categorical_columns_),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.column_transformer_.fit(features)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column_transformer_ is None:
            raise RuntimeError("HousingFeaturePreprocessor must be fitted before transform.")

        features = self._prepare_features(data, fit=False)
        features = self._apply_target_encoding(features)
        transformed = self.column_transformer_.transform(features)
        return pd.DataFrame(
            transformed,
            columns=self.column_transformer_.get_feature_names_out(),
            index=data.index,
        )

    def _prepare_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        features = data.drop(columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS], errors="ignore").copy()

        # Log-transform Size_in_SqFt to align with log-target
        if "Size_in_SqFt" in features.columns:
            features["Size_in_SqFt"] = np.log1p(features["Size_in_SqFt"])

        if AMENITIES_COLUMN in features.columns:
            amenities = features[AMENITIES_COLUMN].fillna("").astype(str)

            if fit:
                values: set[str] = set()
                for row in amenities:
                    values.update(
                        value.strip()
                        for value in row.split(",")
                        if value and value.strip()
                    )
                self.amenity_values_ = sorted(values)

            for amenity in self.amenity_values_:
                column_name = f"amenity_{amenity.lower().replace(' ', '_')}"
                features[column_name] = amenities.apply(
                    lambda value, expected=amenity: int(
                        expected in {item.strip() for item in value.split(",")}
                    )
                )

            features = features.drop(columns=[AMENITIES_COLUMN])

        for column in features.select_dtypes(include=["object", "category"]).columns:
            features[column] = features[column].astype("string")

        return features

    def _apply_target_encoding(self, features: pd.DataFrame) -> pd.DataFrame:
        encoded = features.copy()
        for column in self.high_cardinality_columns_:
            if column not in encoded.columns:
                continue

            mapping = self.target_maps_.get(column, {})
            encoded[f"{column}_target"] = (
                encoded[column].astype("string").map(mapping).fillna(self.global_mean_).astype(float)
            )
            encoded = encoded.drop(columns=[column])

        return encoded


@step
def feature_engineering(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[Any, "preprocessor"],
]:
    """
    Split the raw data and fit preprocessing on the training split only.

    Args:
        df: Raw validated housing dataframe.

    Returns:
        X_train, X_test, y_train, y_test, and the fitted preprocessor artifact.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    X_raw = valid_df.drop(columns=[TARGET_COLUMN])
    y = valid_df[TARGET_COLUMN]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Apply log transformation to the target variable to handle right-skewness
    y_train = np.log1p(y_train_raw)
    y_test = np.log1p(y_test_raw)

    # Use the log-transformed target for fitting the preprocessor (Target Encoding)
    train_raw = pd.concat([X_train_raw, y_train.rename(TARGET_COLUMN)], axis=1)
    test_raw = pd.concat([X_test_raw, y_test.rename(TARGET_COLUMN)], axis=1)

    preprocessor = HousingFeaturePreprocessor().fit(train_raw)
    X_train = preprocessor.transform(train_raw)
    X_test = preprocessor.transform(test_raw)

    print(f"Feature engineering complete. X_train shape: {X_train.shape}")
    print(f"Feature engineering complete. X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor
