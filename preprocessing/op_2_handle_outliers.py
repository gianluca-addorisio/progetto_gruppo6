from typing import NamedTuple
import pandas as pd
import numpy as np

"""Identifies and handles outliers in the dataset.

    - Creates a binary feature for historical buildings (age == 995).
    - Caps extreme values in continuous features using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        HandleOutliersOutputs: The DataFrame with handled outliers.
"""

class HandleOutliersOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_handle_outliers(df: pd.DataFrame) -> HandleOutliersOutputs:
    df_clean = df.copy()
    
    # Special handling for the 'age' column
    if 'age' in df_clean.columns:
        # Create an explicit feature for "historic" buildings
        df_clean['is_historic'] = (df_clean['age'] == 995).astype(int)
        
        # Replace 995 with the maximum normal age
        max_normal_age = df_clean.loc[df_clean['age'] < 995, 'age'].max()
        df_clean.loc[df_clean['age'] == 995, 'age'] = max_normal_age
        print(f"Feature 'is_historic' created. Values age=995 replaced with {max_normal_age}.")

    # IQR method for continuous variables
    cols_to_cap = ['area_percentage', 'height_percentage']
    
    for col in cols_to_cap:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define upper bounds (we do not touch the lower ones because they are physical and > 0)
            upper_bound = Q3 + 1.5 * IQR
            
            # Values beyond upper_bound are lowered to upper_bound
            outliers_count = (df_clean[col] > upper_bound).sum()
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            print(f"Column '{col}': {outliers_count} outliers beyond {upper_bound:.2f} capped.")

    return HandleOutliersOutputs(
        df_output=df_clean
    )