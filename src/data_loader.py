from __future__ import annotations

from pathlib import Path

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
    """Carica un file CSV controllando che il percorso esista."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    return pd.read_csv(path)


def load_raw_data():
    """
    Carica i quattro file originali della competizione.

    Returns:
        train_values, train_labels, test_values, submission_format
    """
    train_values = read_csv(TRAIN_VALUES_FILE)
    train_labels = read_csv(TRAIN_LABELS_FILE)
    test_values = read_csv(TEST_VALUES_FILE)
    submission_format = read_csv(SUBMISSION_FORMAT_FILE)

    return train_values, train_labels, test_values, submission_format


def load_full_train() -> pd.DataFrame:
    """Crea il dataset di training completo unendo feature e target."""
    train_values, train_labels, _, _ = load_raw_data()

    train = train_values.merge(
        train_labels,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    return train


def load_train_test():
    """
    Restituisce X, y, test_values e submission_format.

    X mantiene inizialmente building_id.
    La rimozione dell'identificativo avviene nella fase di preprocessing.
    """
    train = load_full_train()
    _, _, test_values, submission_format = load_raw_data()

    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    return X, y, test_values, submission_format


class DataLoader:
    """
    Classe per caricamento dati e strategie di split.

    Mantiene l'impostazione originale sviluppata su dev, basata su DataLoader
    e split_dataset_by_strategy, ma usa i path centralizzati in src/config.py.
    """

    def __init__(self):
        self.feature_map_df = None

        (
            self.train_values_df,
            self.train_labels_df,
            self.test_values_df,
            self.submission_format_df,
        ) = load_raw_data()

        self.train_df = self.train_values_df.merge(
            self.train_labels_df,
            on=ID_COL,
            how="inner",
            validate="one_to_one",
        )

    def show_info_data(self):
        """Mostra informazioni sulle feature del training set."""
        print("Informazioni sulle feature del training set:")
        self.train_values_df.info()

    def split_dataset_by_strategy(self, choice: int):
        """
        Divide il dataset secondo una strategia scelta.

        Strategie:
            1: Holdout split
            2: Stratified Holdout
            3: K-Fold Cross Validation
            4: Stratified K-Fold
            5: Shuffle Split
            6: Stratified Shuffle Split
        """
        X = self.train_values_df
        y = self.train_labels_df[TARGET_COL]

        # Holdout split
        if choice == 1:
            return train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )

        # Stratified Holdout
        if choice == 2:
            return train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=RANDOM_STATE,
            )

        # K-Fold Cross Validation
        if choice == 3:
            kf = KFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            return list(kf.split(X))

        # Stratified K-Fold
        if choice == 4:
            skf = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            return list(skf.split(X, y))

        # Shuffle Split
        if choice == 5:
            ss = ShuffleSplit(
                n_splits=5,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            return list(ss.split(X))

        # Stratified Shuffle Split
        if choice == 6:
            sss = StratifiedShuffleSplit(
                n_splits=5,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            return list(sss.split(X, y))

        raise ValueError("Scelta non valida. Usa un intero da 1 a 6.")

    def get_group_kfold_splits(self, groups, n_splits: int = 5):
        """
        Restituisce split GroupKFold usando gruppi esterni, ad esempio geografici.
        """
        X = self.train_values_df
        y = self.train_labels_df[TARGET_COL]

        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(X, y, groups=groups))
