import pandas as pd
from op_1_data_evaluation import run_evaluation
from op_2_handle_outliers import run_handle_outliers
from op_3_handle_categorical_features import run_handle_categorical_features
from op_4_scale_numerical_features import run_scale_numerical_features
from op_5_correlation_analysis import run_correlation_analysis
from op__merge_labels import run_merge_labels

"""Executes the full data preprocessing pipeline.

    Args:

    Returns:
        pd.DataFrame: The final preprocessed DataFrame, cleaned and filtered, 
            ready for model development.
"""

def main() -> pd.DataFrame:

    train_values = pd.read_csv("data/raw/train_values.csv")
    test_values = pd.read_csv("data/raw/test_values.csv")
        
    print("\n----- STARTING PIPELINE -----")

    print("Operation 1: Evaluation")
    evaluated_data = run_evaluation(
        train_values
        )
    
    print("Operation 2: Handle Outliers")
    outliers_handled = run_handle_outliers(
        train_values
    )
    
    print("Operation 3: Handle categorical features")
    encoded_data = run_handle_categorical_features(
        outliers_handled.df_output
    )

    print("Operation 4: Scale Numerical Features")
    scaled_data = run_scale_numerical_features(
        test_values,
        encoded_data.df_output
    )

    train_scaled = scaled_data.train_scaled
    test_scaled = scaled_data.test_scaled

    print("Operation 5: Correlation Analysis")
    uncorrelated_data = run_correlation_analysis(
        train_scaled,
    )

    print("Operation : Merge Labels and Clean Target")
    scaled_data = run_correlation_analysis(
        uncorrelated_data.df_output,
    )

    print("Final Dataset ready:")
    # final_dataset =
    # print(final_dataset.head)

    print("\n----- PIPELINE COMPLETED SUCCESSFULLY -----")
    
    return
    
if __name__ == "__main__":
    main()