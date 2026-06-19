from __future__ import annotations

import pandas as pd


GEO_FEATURES = [
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id",
]

NUMERIC_FEATURES = [
    "count_floors_pre_eq",
    "age",
    "count_families",
    "building_volume_proxy",
    "total_secondary_use_count",
    "total_superstructure_count",
    "has_fragile_material",
    "has_engineered_structure",
]

CATEGORICAL_FEATURES = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
]

FRAGILE_MATERIAL_FEATURES = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_mud_mortar_brick",
]

ENGINEERED_STRUCTURE_FEATURES = [
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
]


def get_superstructure_features(df: pd.DataFrame) -> list[str]:
    """Return binary features related to structural materials."""
    return [col for col in df.columns if col.startswith("has_superstructure_")]


def get_secondary_use_features(df: pd.DataFrame) -> list[str]:
    """Return binary features related to secondary building use."""
    return [
        col for col in df.columns
        if col.startswith("has_secondary_use_") and col != "has_secondary_use"
    ]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compact and interpretable engineered features.

    The final pruning of candidate or redundant features is handled later by
    DataCleaner inside the preprocessing pipeline.
    """
    df = df.copy()

    superstructure_cols = get_superstructure_features(df)
    secondary_use_cols = get_secondary_use_features(df)

    if superstructure_cols:
        df["total_superstructure_count"] = df[superstructure_cols].sum(axis=1)

    if secondary_use_cols:
        df["total_secondary_use_count"] = df[secondary_use_cols].sum(axis=1)

    existing_fragile_cols = [
        col for col in FRAGILE_MATERIAL_FEATURES if col in df.columns
    ]
    if existing_fragile_cols:
        df["has_fragile_material"] = (
            df[existing_fragile_cols].sum(axis=1) > 0
        ).astype(int)

    existing_engineered_cols = [
        col for col in ENGINEERED_STRUCTURE_FEATURES if col in df.columns
    ]
    if existing_engineered_cols:
        df["has_engineered_structure"] = (
            df[existing_engineered_cols].sum(axis=1) > 0
        ).astype(int)

    if {"area_percentage", "height_percentage"}.issubset(df.columns):
        df["building_volume_proxy"] = (
            df["area_percentage"] * df["height_percentage"]
        )

    return df


def get_feature_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of the main feature groups."""
    rows = [
        {
            "group": "geographical",
            "n_features": len([c for c in GEO_FEATURES if c in df.columns]),
        },
        {
            "group": "numeric_and_aggregated",
            "n_features": len([c for c in NUMERIC_FEATURES if c in df.columns]),
        },
        {
            "group": "categorical_structural",
            "n_features": len([c for c in CATEGORICAL_FEATURES if c in df.columns]),
        },
        {
            "group": "superstructure_binary_original",
            "n_features": len(get_superstructure_features(df)),
        },
        {
            "group": "secondary_use_binary_original",
            "n_features": len(get_secondary_use_features(df)),
        },
    ]

    return pd.DataFrame(rows)
