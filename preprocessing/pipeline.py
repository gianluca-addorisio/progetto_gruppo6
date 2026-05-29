import pandas as pd
from op_1_data_evaluation import run_evaluation
from op_2_handle_categorical_features import run_handle_categorical_features

"""Executes the full data preprocessing pipeline.

    Args:

    Returns:
        pd.DataFrame: The final preprocessed DataFrame, cleaned and filtered, 
            ready for model development.
"""

def main() -> pd.DataFrame:

    train_values = pd.read_csv("data/raw/train_values.csv")
        
    print("\n----- STARTING PIPELINE -----")

    print("Operation 1: Evaluation")
    evaluated_data = run_evaluation(
        train_values
        )
    
    print("Operation 2: Handle categorical features")
    only_numerical_data = run_handle_categorical_features(
        train_values
    )

    print("Operation 3:")
    only_numerical_data = run_handle_categorical_features(
        train_values
    )



    print("Final Dataset ready:")
    # final_dataset =
    # print(final_dataset.head)

    print("\n----- PIPELINE COMPLETED SUCCESSFULLY -----")
    
    return
    
if __name__ == "__main__":
    main()