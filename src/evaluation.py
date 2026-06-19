"""
Utility per la valutazione dei modelli.

Il modulo raccoglie le funzioni usate per calcolare le metriche principali del
progetto e per salvare report diagnostici. La metrica di riferimento è la
micro-F1, coerente con la valutazione della competizione, mentre macro-F1 e
weighted-F1 vengono mantenute per analisi comparative più complete.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true, y_pred, model_name: str) -> dict[str, object]:
    """
    Calcola le principali metriche di valutazione del progetto.

    La micro-F1 è la metrica ufficiale di riferimento. Macro-F1 e weighted-F1
    vengono salvate per rendere più informativo il confronto tra modelli.
    """
    return {
        "model": model_name,
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def get_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Restituisce il classification report come DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def save_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Salva su disco la matrice di confusione del modello."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
