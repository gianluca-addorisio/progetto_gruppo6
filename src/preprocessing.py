from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COL, RANDOM_STATE, TARGET_COL
from .data_loader import DataLoader
from .features import (
    CATEGORICAL_FEATURES,
    add_engineered_features,
)


FINAL_DROP_FEATURES = [
    # Identifiers / target
    ID_COL,
    TARGET_COL,

    # Replaced by building_volume_proxy
    "area_percentage",
    "height_percentage",

    # Exploratory age/group features not used in the final compact feature set
    "age_clipped",
    "age_group",
    "family_count_group",
    "floor_count_group",

    # Low-informative categorical features removed from the final feature set
    "plan_configuration",
    "legal_ownership_status",
]

GEO_ONE_HOT_FEATURES = [
    "geo_level_1_id",
]

GEO_FREQUENCY_FEATURES = [
    "geo_level_2_id",
    "geo_level_3_id",
]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoder for high-cardinality categorical identifiers.

    The encoder is fitted only on the training data inside the sklearn pipeline.
    During transform, unseen categories are mapped to 0.0.
    """

    def __init__(self):
        self.frequency_maps_: dict[str, pd.Series] = {}
        self.input_features_: list[str] = []

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.input_features_ = list(X_df.columns)
        self.frequency_maps_ = {
            col: X_df[col].value_counts(normalize=True)
            for col in self.input_features_
        }
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X, columns=self.input_features_)

        encoded = pd.DataFrame(index=X_df.index)

        for col in self.input_features_:
            freq_map = self.frequency_maps_[col]
            encoded[f"{col}_freq"] = (
                X_df[col]
                .map(freq_map)
                .fillna(0.0)
                .astype(float)
            )

        return encoded

    def get_feature_names_out(self, input_features=None):
        features = input_features if input_features is not None else self.input_features_
        return np.array([f"{col}_freq" for col in features], dtype=object)

    @staticmethod
    def _to_dataframe(X, columns=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()

        return pd.DataFrame(X, columns=columns)


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with different scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def get_original_secondary_use_features(df: pd.DataFrame) -> list[str]:
    """Return original secondary-use binary features to remove."""
    return [
        col for col in df.columns
        if col == "has_secondary_use" or col.startswith("has_secondary_use_")
    ]


def get_original_superstructure_features(df: pd.DataFrame) -> list[str]:
    """Return original superstructure binary features to remove."""
    return [
        col for col in df.columns
        if col.startswith("has_superstructure_")
    ]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw feature data for modeling.

    Operations:
        - remove identifiers and target if present;
        - add selected engineered features;
        - remove obsolete, redundant or compressed original features.
    """
    df = df.copy()

    df = add_engineered_features(df)

    dynamic_drop_features = (
        get_original_secondary_use_features(df)
        + get_original_superstructure_features(df)
    )

    cols_to_drop = [
        col for col in FINAL_DROP_FEATURES + dynamic_drop_features
        if col in df.columns
    ]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Create a stratified train/validation split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def infer_preprocessing_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Infer columns for final preprocessing.

    Final geographic strategy:
        - geo_level_1_id: one-hot encoding;
        - geo_level_2_id and geo_level_3_id: frequency encoding.

    Other low-cardinality categorical features are one-hot encoded.
    Numeric and aggregate features are passed through or scaled depending on
    the selected model pipeline.
    """
    categorical_cols = [
        col for col in CATEGORICAL_FEATURES + GEO_ONE_HOT_FEATURES
        if col in X.columns
    ]

    dtype_categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_cols = sorted(set(categorical_cols + dtype_categorical))

    geo_frequency_cols = [
        col for col in GEO_FREQUENCY_FEATURES
        if col in X.columns
    ]

    excluded_from_numeric = set(categorical_cols + geo_frequency_cols)

    numeric_cols = [
        col for col in X.columns
        if col not in excluded_from_numeric
        and pd.api.types.is_numeric_dtype(X[col])
    ]

    return categorical_cols, geo_frequency_cols, numeric_cols


def infer_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Backward-compatible helper returning categorical and numeric columns.

    For the final pipeline, use infer_preprocessing_columns instead.
    """
    categorical_cols, geo_frequency_cols, numeric_cols = infer_preprocessing_columns(X)
    numeric_with_frequency_inputs_removed = [
        col for col in numeric_cols
        if col not in geo_frequency_cols
    ]
    return categorical_cols, numeric_with_frequency_inputs_removed


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build a preprocessing transformer for the final compact feature matrix.
    """
    categorical_cols, geo_frequency_cols, numeric_cols = infer_preprocessing_columns(X)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    transformers = []

    if categorical_cols:
        transformers.append(
            ("categorical", make_one_hot_encoder(), categorical_cols)
        )

    if geo_frequency_cols:
        transformers.append(
            ("geo_frequency", FrequencyEncoder(), geo_frequency_cols)
        )

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_transformer, numeric_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def make_model_pipeline(
    model,
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessing + model.
    """
    preprocessor = build_preprocessor(
        X=X,
        scale_numeric=scale_numeric,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def preprocess():
    """
    Load training data and return the prepared compact feature matrix.
    """
    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    X_prepared = prepare_features(X)
    return X_prepared, y


if __name__ == "__main__":
    X, y = preprocess()
    print(X.shape)
    print(y.shape)
