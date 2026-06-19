"""
Costruzione della pipeline di preprocessing e modellazione.

Il modulo definisce l'ordine delle trasformazioni applicate ai dati prima del
fit del modello: feature engineering, pulizia, gestione dell'età, encoding
geografico, encoding categorico e scaling opzionale. La funzione
make_complete_pipeline aggiunge poi eventuale feature selection, eventuale PCA
e stimatore finale in un'unica pipeline sklearn.
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..config import RANDOM_STATE
from ..features import add_engineered_features
from .age_handler import AgeHandler
from .cleaner import DataCleaner
from .encoding import CategoricalEncoder, FrequencyEncoder
from .scaling import NumericalScaler


def get_preprocessing_steps(scale_numeric: bool = False):
    """
    Restituisce la lista ordinata degli step di preprocessing.

    Lo scaling numerico è opzionale perché non è necessario per i modelli ad
    alberi, ma diventa utile quando si applica PCA o quando si usano modelli
    sensibili alla scala delle variabili.
    """

    steps = [
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("cleaner", DataCleaner()),
        ("age_handler", AgeHandler()),
        ("geo_freq_encoder", FrequencyEncoder()),
        ("cat_encoder", CategoricalEncoder()),
    ]

    if scale_numeric:
        steps.append(("scaler", NumericalScaler()))

    return steps


def get_preprocessing_pipeline(scale_numeric: bool = False):
    """Costruisce una pipeline composta solo dagli step di preprocessing."""
    return Pipeline(get_preprocessing_steps(scale_numeric))


def make_complete_pipeline(
    model,
    scale_numeric: bool = False,
    feature_selector=None,
    use_pca: bool = False,
    pca_n_components: int = 40,
):
    """
    Costruisce la pipeline completa di preprocessing, selezione feature e modello.

    La funzione mantiene nello stesso oggetto sklearn tutte le trasformazioni
    apprese dai dati, riducendo il rischio di incoerenza tra validazione,
    training finale e inferenza sul test set.
    """
    if use_pca:
        scale_numeric = True

    steps = get_preprocessing_steps(scale_numeric)

    # Il feature selector viene inserito dentro la pipeline per evitare data
    # leakage durante validazione e tuning.
    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))

    if use_pca:
        steps.append(
            ("pca", PCA(n_components=pca_n_components, random_state=RANDOM_STATE))
        )

    steps.append(("model", model))

    return Pipeline(steps)
