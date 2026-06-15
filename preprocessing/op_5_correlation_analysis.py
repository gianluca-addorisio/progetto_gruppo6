from typing import NamedTuple
import pandas as pd
import numpy as np

"""Analyzes feature correlation and drops highly correlated features.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        CorrelationOutputs: The DataFrame with reduced multicollinearity.
"""

class CorrelationOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_correlation_analysis(df: pd.DataFrame) -> CorrelationOutputs:
    # Set the correlation threshold
    threshold = 0.85
        
    # Calculate the absolute correlation matrix for numerical columns only
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
    
    # Select only the upper triangle of the correlation matrix to avoid duplicates
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop based on the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Found {len(to_drop)} highly correlated features to remove:")
        print(to_drop)
        df_reduced = df.drop(columns=to_drop)
    else:
        print("No features exceeded the correlation threshold.")
        df_reduced = df.copy()
        
    return CorrelationOutputs(
        df_output=df_reduced
    )