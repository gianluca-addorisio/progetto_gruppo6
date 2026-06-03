from typing import NamedTuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""Scales numerical features for both train and test datasets.

    To prevent data leakage, the scaler is fitted ONLY on the training data.
    Then, both the training and test data are transformed using this fitted scaler.

    Args:
        df_train (pd.DataFrame): The encoded training DataFrame.
        df_test (pd.DataFrame): The encoded testing DataFrame.
    
    Returns:
        ScaleNumericalOutputs: Contains both the scaled train and test DataFrames.
"""

class ScaleNumericalOutputs(NamedTuple):
    train_scaled: pd.DataFrame
    test_scaled: pd.DataFrame

def run_scale_numerical_features(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame
) -> ScaleNumericalOutputs:
    
    train_output = df_train.copy()
    test_output = df_test.copy()
    
    # Define continuous numerical columns to scale
    # We exclude geo_level_ids (often treated as categorical/embedding) and binary flags
    numerical_cols = [
        'count_floors_pre_eq', 
        'age', 
        'area_percentage', 
        'height_percentage', 
        'count_families'
    ]
    
    scaler = StandardScaler()
    
    # FIT e TRANSFORM on Train Values
    train_output[numerical_cols] = scaler.fit_transform(train_output[numerical_cols])
    
    # Only TRANSFORM on Test Values (using the parameters calculated on the train set)
    test_output[numerical_cols] = scaler.transform(test_output[numerical_cols])


    return ScaleNumericalOutputs(
        train_scaled=train_output,
        test_scaled=test_output
    )