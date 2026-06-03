from typing import NamedTuple
import pandas as pd

"""Performs a comprehensive exploratory data analysis on a DataFrame.

    This function prints a summary of the dataset, including information about 
    data types, missing values, statistical descriptions, and unique values 
    for categorical columns. It helps identify data quality issues before 
    proceeding to the cleaning phase.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        df_output (pd.DataFrame): The original DataFrame.
"""

class EvaluationOutputs(NamedTuple):
    df_output: pd.DataFrame

def run_evaluation(df: pd.DataFrame) -> EvaluationOutputs :
    print("DataFrame Information Summary:")
    # Provides a summary of the dataset information
    df.info()
    print("\n" + "="*50 + "\n")

    print("Missing Values Analysis:")
    # Counts how many records are null in each column
    null_values_per_column = df.isnull().sum()
    # Identifies columns with null values
    columns_with_nulls = null_values_per_column[null_values_per_column > 0]
    
    if columns_with_nulls.empty:
        # If the series is empty, it means no nulls were found
        print("No missing values were found in the dataset.")
    else:
        # If not empty, print the count for each column affected
        print("Columns with missing values:")
        print(columns_with_nulls)

    print("\n" + "="*50 + "\n")

    print("Statistical Description of the DataFrame:")
    # Prints statistical data for both numerical and categorical columns
    print(df.describe(include='all'))
    print("\n" + "="*50 + "\n")

    # Print a summary of the dataset's structure, 
    # including its dimensions, data types, and the total count of missing values.
    # This provides a quick overview to ensure the data was loaded correctly 
    # before proceeding with cleaning.
    print("Summary:")
    shape = df.shape
    data_types = df.dtypes.unique()
    print(f"""The DataFrame contains {shape[1]} columns and {shape[0]} rows.
            The possible data types are: {data_types}.
            Total null values found: {columns_with_nulls.sum()}.""")

    print("\n" + "="*50 + "\n")

    print("First 5 rows preview:")
    # Prints the first rows of the dataframe (default is 5)
    print(df.head())
    print("\n" + "="*50 + "\n")

    # List of columns with dtype 'O' (object)
    object_columns = df.select_dtypes(include=['object']).columns

    print("Object-type Columns Analysis:")
    if object_columns.empty:
        print("No object-type (string) columns found in the dataset.")
    else:
        # These columns must be processed because the ML models requires numeric input
        print(f"Found {len(object_columns)} object-type columns: {list(object_columns)}")
        
        df_objects = df[object_columns]
        
        
        print("\nFirst rows of the Object-only DataFrame:")
        print(df_objects.head())

        # This is done to inspect the actual values of these columns, 
        # to determine the best cleaning strategy
        print("\nUnique values within Object columns:")
        for col in object_columns:
            unique_values = df_objects[col].unique()

            print(f"\nColumn: **{col}**")
            print(f"Values: {unique_values}")

    print("\n" + "="*50 + "\n")

    return EvaluationOutputs(
        df_output = df
    )