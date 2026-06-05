from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Importiamo i nostri moduli personalizzati
from .cleaner import DataCleaner
from .age_handler import AgeHandler

# from .outliers import OutlierCapper Rimossa perchè 
from .encoding import CategoricalEncoder, FrequencyEncoder
from .scaling import NumericalScaler
from sklearn.decomposition import PCA

# Importiamo la funzione per il feature engineering esistente
from ..features import add_engineered_features

def get_preprocessing_steps(scale_numeric=False):
    """
    Restituisce la lista dei passi di preprocessing aggiornata alla nuova strategia.
    
    L'ordine è critico:
    1. Feature Engineering: crea nuove feature da quelle esistenti.
    2. DataCleaner: rimuove le originali ormai ridondanti (e identificativi).
    3. AgeHandler: gestisce il valore anomalo di age = 995.
    4. FrequencyEncoder: encoding specifico per geo_level_2 e geo_level_3.
    5. CategoricalEncoder: One-Hot per le categoriche e geo_level_1_id.
    6. NumericalScaler (opzionale): scaling finale quando richiesto.
    """
    steps = [
        ('feature_engineering', FunctionTransformer(add_engineered_features)),
        ('cleaner', DataCleaner()),
        ('age_handler', AgeHandler()),
        # OutlierCapper è stato rimosso nella pipeline standard perchè le colonne su cui agisce vengono già rimosse dal DataCleaner.
        ('geo_freq_encoder', FrequencyEncoder()),
        ('cat_encoder', CategoricalEncoder()),
    ]
    
    if scale_numeric:
        steps.append(('scaler', NumericalScaler()))
        
    return steps

def get_preprocessing_pipeline(scale_numeric=False):
    """
    Costruisce la pipeline di preprocessing completa.
    """
    return Pipeline(get_preprocessing_steps(scale_numeric))

def make_complete_pipeline(model, scale_numeric=False, feature_selector=None, use_pca=False, pca_n_components: int = 40):
    """
    Crea una pipeline "piatta" (non nidificata) che include sia 
    preprocessing che modello.
    """
    if use_pca:
        scale_numeric = True

    steps = get_preprocessing_steps(scale_numeric)
    
    if feature_selector is not None:
        steps.append(('feature_selector', feature_selector))

    if use_pca:
        steps.append(('pca', PCA(n_components=pca_n_components, random_state=42)))

    steps.append(('model', model))
    
    return Pipeline(steps)
