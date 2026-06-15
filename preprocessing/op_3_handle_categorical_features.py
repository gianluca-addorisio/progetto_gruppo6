from typing import NamedTuple
import pandas as pd

class HandleCategoricalFeaturesOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_handle_categorical_features(df: pd.DataFrame) -> HandleCategoricalFeaturesOutputs:
    
    # Select all 'object' type columns (the 8 categorical columns found in the previous step)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply One-Hot Encoding
    # drop_first=True removes the first generated column for each category 
    # to avoid multicollinearity issues
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Convert all boolean columns (produced by .get_dummies) to integer (True=1, False=0)
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

    # Identify the columns that were added by the encoding
    original_categorical_cols = [
        'land_surface_condition', 'foundation_type', 'roof_type', 
        'ground_floor_type', 'other_floor_type', 'position', 
        'plan_configuration', 'legal_ownership_status'
    ]

    # Print the new columns and a preview of their values
    print("Newly Created Encoded Columns")
    encoded_cols = [col for col in df_encoded.columns if any(cat in col for cat in original_categorical_cols)]
    print(df_encoded[encoded_cols].head())

    print("\nSummary of New Feature Count")
    print(f"Total columns before encoding: {df.shape[1]}")
    print(f"Total columns after encoding: {df_encoded.shape[1]}")

    return HandleCategoricalFeaturesOutputs(
        df_output=df_encoded
    )