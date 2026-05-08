from typing import Annotated, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
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
    """Train-fitted preprocessor using K-Fold Target Encoding to prevent leakage."""

    def __init__(self) -> None:
        self.amenity_values_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.high_cardinality_columns_: list[str] = []
        self.target_maps_: dict[str, dict[str, float]] = {}
        self.global_mean_: float = 0.0
        self.column_transformer_: ColumnTransformer | None = None

    def fit(self, data: pd.DataFrame) -> "HousingFeaturePreprocessor":
        # data must contain TARGET_COLUMN (log-scale)
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
        
        # Fit final maps for transform (inference) using all training data
        self.target_maps_ = {}
        for column in self.high_cardinality_columns_:
            stats = data.groupby(column)[TARGET_COLUMN].agg(["count", "mean"])
            smoothed = (
                stats["count"] * stats["mean"] + TARGET_ENCODING_SMOOTHING * self.global_mean_
            ) / (stats["count"] + TARGET_ENCODING_SMOOTHING)
            self.target_maps_[column] = smoothed.to_dict()
            
        return self

    def fit_transform_kfold(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform training data using K-Fold to prevent target leakage.
        """
        self.fit(data)
        features = self._prepare_features(data, fit=False)
        encoded = features.copy()
        
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        for column in self.high_cardinality_columns_:
            # Initialize with global mean
            encoded[f"{column}_target"] = self.global_mean_
            
            for train_idx, val_idx in kf.split(features):
                # Calculate means using only the 4 training folds
                fold_train = data.iloc[train_idx]
                stats = fold_train.groupby(column)[TARGET_COLUMN].agg(["count", "mean"])
                
                smoothed = (
                    stats["count"] * stats["mean"] + TARGET_ENCODING_SMOOTHING * self.global_mean_
                ) / (stats["count"] + TARGET_ENCODING_SMOOTHING)
                
                # Apply to the 1 validation fold
                mapping = smoothed.to_dict()
                val_categories = encoded.iloc[val_idx][column]
                encoded.loc[encoded.index[val_idx], f"{column}_target"] = (
                    val_categories.map(mapping).fillna(self.global_mean_)
                )
            
            encoded = encoded.drop(columns=[column])

        # Apply standard pipelines (Scaling, OneHot)
        self.numeric_columns_ = encoded.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_columns_ = [c for c in encoded.columns if c not in self.numeric_columns_]
        
        # Refit column transformer on the OOF-encoded features
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _make_one_hot_encoder())])
        
        self.column_transformer_ = ColumnTransformer([
            ("numeric", numeric_pipeline, self.numeric_columns_),
            ("categorical", categorical_pipeline, self.categorical_columns_),
        ], remainder="drop", verbose_feature_names_out=False)
        
        transformed = self.column_transformer_.fit_transform(encoded)
        return pd.DataFrame(transformed, columns=self.column_transformer_.get_feature_names_out(), index=data.index)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(data, fit=False)
        
        # Apply pre-learned maps (Inference mode)
        for column in self.high_cardinality_columns_:
            mapping = self.target_maps_.get(column, {})
            features[f"{column}_target"] = features[column].map(mapping).fillna(self.global_mean_).astype(float)
            features = features.drop(columns=[column])
            
        transformed = self.column_transformer_.transform(features)
        return pd.DataFrame(transformed, columns=self.column_transformer_.get_feature_names_out(), index=data.index)

    def _prepare_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        features = data.drop(columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS], errors="ignore").copy()

        # Strip whitespace and log-transform size
        for col in features.select_dtypes(include=["object", "string"]).columns:
            features[col] = features[col].astype(str).str.strip()

        if "Size_in_SqFt" in features.columns:
            features["Size_in_SqFt"] = np.log1p(features["Size_in_SqFt"])

        if AMENITIES_COLUMN in features.columns:
            amenities = features[AMENITIES_COLUMN].fillna("").astype(str)
            if fit:
                values = set()
                for row in amenities:
                    values.update(v.strip() for v in row.split(",") if v.strip())
                self.amenity_values_ = sorted(values)

            for amenity in self.amenity_values_:
                features[f"amenity_{amenity.lower().replace(' ', '_')}"] = amenities.apply(
                    lambda v, exp=amenity: int(exp in {i.strip() for i in v.split(",")})
                )
            features = features.drop(columns=[AMENITIES_COLUMN])

        for column in features.select_dtypes(include=["object", "category"]).columns:
            features[column] = features[column].astype("string")
        return features


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
    Split raw data and fit K-Fold Target Encoding on the training split.
    """
    valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    X_raw = valid_df.drop(columns=[TARGET_COLUMN])
    y = valid_df[TARGET_COLUMN]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_train = np.log1p(y_train_raw)
    y_test = np.log1p(y_test_raw)

    train_raw = pd.concat([X_train_raw, y_train.rename(TARGET_COLUMN)], axis=1)
    
    preprocessor = HousingFeaturePreprocessor()
    X_train = preprocessor.fit_transform_kfold(train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"Feature engineering complete. X_train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test, preprocessor
