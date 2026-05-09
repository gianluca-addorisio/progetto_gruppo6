from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from .config import (
    ID_COL,
    RANDOM_STATE,
    SUBMISSION_FORMAT_FILE,
    TARGET_COL,
    TEST_VALUES_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_VALUES_FILE,
)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the four original competition files.

    Returns:
        train_values, train_labels, test_values, submission_format
    """
    train_values = read_csv(TRAIN_VALUES_FILE)
    train_labels = read_csv(TRAIN_LABELS_FILE)
    test_values = read_csv(TEST_VALUES_FILE)
    submission_format = read_csv(SUBMISSION_FORMAT_FILE)

    return train_values, train_labels, test_values, submission_format


def load_full_train() -> pd.DataFrame:
    """
    Load the full training dataset by merging features and labels on building_id.
    """
    train_values, train_labels, _, _ = load_raw_data()

    train = train_values.merge(
        train_labels,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    return train


def load_train_test() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load train features, target, test features and submission format.

    Returns:
        X: training features, including building_id
        y: target series
        test_values: test features, including building_id
        submission_format: submission template
    """
    train = load_full_train()
    _, _, test_values, submission_format = load_raw_data()

    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    return X, y, test_values, submission_format


def split_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Create a simple holdout train/validation split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def split_stratified_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Create a stratified holdout train/validation split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def get_kfold_splits(
    X: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    """Return K-Fold split indices."""
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(X))


def get_stratified_kfold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    """Return Stratified K-Fold split indices."""
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(X, y))


def get_shuffle_split_splits(
    X: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Return ShuffleSplit indices."""
    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    return list(splitter.split(X))


def get_stratified_shuffle_split_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Return StratifiedShuffleSplit indices."""
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    return list(splitter.split(X, y))


def get_group_kfold_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups,
    n_splits: int = 5,
):
    """
    Return GroupKFold split indices.

    This is useful if the group structure, for example geographical groups,
    should be kept separated across folds.
    """
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(X, y, groups=groups))


class DataLoader:
    """
    Backward-compatible wrapper around the functional API.

    This preserves the original idea from the dev branch while exposing cleaner
    reusable functions for notebooks and scripts.
    """

    def __init__(self):
        self.train_values_df, self.train_labels_df, self.test_values_df, self.submission_format_df = load_raw_data()
        self.train_df = load_full_train()

    def show_info_data(self) -> None:
        """Print information about the training feature dataset."""
        print("Training features info:")
        self.train_values_df.info()

    def split_dataset_by_strategy(self, choice: int):
        """
        Return split according to a numeric strategy.

        Choices:
            1: holdout
            2: stratified holdout
            3: K-Fold
            4: Stratified K-Fold
            5: ShuffleSplit
            6: StratifiedShuffleSplit
        """
        X = self.train_values_df
        y = self.train_labels_df[TARGET_COL]

        if choice == 1:
            return split_holdout(X, y)

        if choice == 2:
            return split_stratified_holdout(X, y)

        if choice == 3:
            return get_kfold_splits(X)

        if choice == 4:
            return get_stratified_kfold_splits(X, y)

        if choice == 5:
            return get_shuffle_split_splits(X)

        if choice == 6:
            return get_stratified_shuffle_split_splits(X, y)

        raise ValueError("Invalid choice. Use an integer from 1 to 6.")
