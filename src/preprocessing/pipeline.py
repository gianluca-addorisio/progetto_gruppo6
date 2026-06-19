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
    Return the preprocessing steps used by the final training pipeline.

    Order:
    1. Feature engineering from raw input variables.
    2. Cleaning and removal of redundant or identifier columns.
    3. Handling of anomalous building-age values.
    4. Frequency encoding for high-cardinality geographical variables.
    5. One-hot encoding for categorical variables.
    6. Optional numerical scaling, mainly required before PCA.
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
    """Build a preprocessing-only pipeline."""
    return Pipeline(get_preprocessing_steps(scale_numeric))


def make_complete_pipeline(
    model,
    scale_numeric: bool = False,
    feature_selector=None,
    use_pca: bool = False,
    pca_n_components: int = 40,
):
    """
    Build the complete modeling pipeline.

    The pipeline applies preprocessing first, then optional feature selection,
    optional PCA and finally the estimator.
    """
    if use_pca:
        scale_numeric = True

    steps = get_preprocessing_steps(scale_numeric)

    if feature_selector is not None:
        steps.append(("feature_selector", feature_selector))

    if use_pca:
        steps.append(
            ("pca", PCA(n_components=pca_n_components, random_state=RANDOM_STATE))
        )

    steps.append(("model", model))

    return Pipeline(steps)
