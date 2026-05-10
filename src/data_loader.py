import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    GroupKFold
)
from pathlib import Path

class DataLoader:

    def __init__(self):
        self.feature_map_df = None
        self.__train_values_df = pd.read_csv('../data/raw/train_values.csv')
        self.__train_labels_df = pd.read_csv('../data/raw/train_labels.csv')
        self.train_df = self.__train_values_df.merge(self.__train_labels_df, on="building_id")

    def show_info_data(self):
        print("Here's the features of our training data and their stats:")
        self.__train_values_df.info()

    def split_dataset_by_strategy(self, choice: int):

        X = self.__train_values_df
        y = self.__train_labels_df["damage_grade"]

        #Holdout split
        if choice == 1:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            return X_train, X_validation, y_train, y_validation
        #Stratified Holdout
        elif choice == 2:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )
            return X_train, X_validation, y_train, y_validation
        #K-Fold Cross Validation
        elif choice == 3:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            return list(kf.split(X))
        #Stratified K-Fold
        elif choice == 4:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return list(skf.split(X, y))
        #Shuffle Split
        elif choice == 5:
            ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            return list(ss.split(X))
        #Stratified Shuffle Split
        elif choice == 6:
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            return list(sss.split(X, y))
        else:
            raise ValueError("Invalid choice. Use 1-7.")

