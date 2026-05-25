from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data_loader import DataLoader
from .feature_selection import FeatureSelection

from .config import ID_COL, RANDOM_STATE, TARGET_COL
from .features import (
    CATEGORICAL_FEATURES,
    GEO_FEATURES,
    add_engineered_features,
)


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with different scikit-learn versions.

    Newer versions use sparse_output, older versions use sparse. Naturally,
    the parameter name changed because apparently naming things once was too
    peaceful.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw feature data for modeling.

    Operations:
        - remove building_id;
        - remove target if accidentally present;
        - add engineered features.
    """
    df = df.copy()

    cols_to_drop = [col for col in [ID_COL, TARGET_COL] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = add_engineered_features(df)

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


def infer_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Infer categorical and numeric columns for preprocessing.

    Geographical IDs are treated as categorical variables, not continuous
    numerical quantities.
    """
    categorical_candidates = CATEGORICAL_FEATURES + GEO_FEATURES

    explicit_categorical = [
        col for col in categorical_candidates if col in X.columns
    ]

    dtype_categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_cols = sorted(set(explicit_categorical + dtype_categorical))

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(X[col])
    ]

    return categorical_cols, numeric_cols


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build a preprocessing transformer for a given feature matrix.

    Args:
        X: feature matrix after prepare_features.
        scale_numeric: whether to apply StandardScaler to numeric columns.
    """
    categorical_cols, numeric_cols = infer_column_groups(X)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
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

    data_loader = DataLoader()
    X, y = data_loader.load_train_test()
    X_prepared = prepare_features(X)
    #scegliere che split applicare
    #data_loader.split_dataset_by_strategy(3,X_prepared, y)
    return X_prepared, y


if __name__ == "__main__":

    X, y = preprocess()
    fs = FeatureSelection()
    #ranking = fs.correlation_matrix(X, y)
    #print(ranking)

    #x_selected = fs.Relief_selection(X,y)


    #final_df_chi2 = fs.chi_square_selection(X, y)

    #print(final_df.head())
    #final_df_MU = fs.information_gain_selection(X, y)
    #print(final_df_MU.head())
    final_df_random_forest = fs.random_forest_selection(X, y)
    print(final_df_random_forest.head())
    #latent_df = fs.autoencoder_selection(X)

    #print(latent_df.head())

    df_xgb = fs.xgboost_selection(X, y)

    df_cat = fs.catboost_selection(X, y)