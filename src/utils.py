from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> None:
    """Save a DataFrame as CSV, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_model(model, path: str | Path) -> None:
    """Save a fitted model using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Load a model saved with joblib."""
    return joblib.load(path)
