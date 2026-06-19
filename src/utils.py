"""
Utility generali per salvataggio, caricamento e visualizzazione.

Il modulo raccoglie funzioni di supporto usate in più punti del progetto:
creazione di directory, salvataggio di DataFrame, serializzazione dei modelli
e generazione di grafici diagnostici per ranking delle feature e correlazioni.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR


def ensure_dir(path: str | Path) -> Path:
    """Crea una directory se non esiste e restituisce il percorso come Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> None:
    """Salva un DataFrame in formato CSV creando le cartelle necessarie."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_model(model, path: str | Path) -> None:
    """Salva un modello addestrato tramite joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Carica un modello precedentemente salvato tramite joblib."""
    return joblib.load(path)


def plot_feature_ranking(
    scores: pd.Series,
    title: str = "Feature Ranking",
    save_path: str | Path = FIGURES_DIR / "feature_ranking.png",
) -> None:
    """Salva su disco il grafico di ranking delle feature."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    scores = scores.sort_values(ascending=False)

    plt.figure(figsize=(20, 12))
    plt.barh(scores.index, scores.values)
    plt.gca().invert_yaxis()
    plt.xlabel("Score")
    plt.ylabel("Features")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    save_path: str | Path = FIGURES_DIR / "correlation_heatmap.png",
) -> None:
    """Salva su disco la heatmap delle correlazioni numeriche."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(24, 16))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()