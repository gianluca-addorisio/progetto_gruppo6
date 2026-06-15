from typing import NamedTuple
import pandas as pd

"""Merges the training values with their corresponding labels and drops IDs.

    Args:
        df_values (pd.DataFrame): The preprocessed training values.
        labels_path (str): The file path to the target labels CSV.
    
    Returns:
        MergeLabelsOutputs: The final DataFrame ready for training.
"""

class MergeLabelsOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_merge_labels(df_values: pd.DataFrame, labels_path: str) -> MergeLabelsOutputs:
    # Read the labels dataset
    df_labels = pd.read_csv(labels_path)
    
    # Merge values and labels on 'building_id'
    df_merged = pd.merge(df_values, df_labels, on='building_id')
    
    # Drop 'building_id' as it's an identifier and should not be used as a predictive feature
    df_merged = df_merged.drop(columns=['building_id'])
    
    print(f"Final dataset shape: {df_merged.shape}")
    print(f"Target column 'damage_grade' value counts:\n{df_merged['damage_grade'].value_counts(normalize=True)}")
    
    return MergeLabelsOutputs(
        df_output=df_merged
    )