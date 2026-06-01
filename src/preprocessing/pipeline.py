from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Importiamo i nostri moduli personalizzati
from .cleaner import DataCleaner
from .age_handler import AgeHandler
from .outliers import OutlierCapper
from .encoding import CategoricalEncoder, FrequencyEncoder
from .scaling import NumericalScaler

# Importiamo la funzione per il feature engineering esistente
from ..features import add_engineered_features

def get_preprocessing_steps(scale_numeric=True):
    """
    Restituisce la lista dei passi di preprocessing aggiornata alla nuova strategia.
    
    L'ordine è critico:
    1. Feature Engineering: crea nuove feature da quelle esistenti.
    2. DataCleaner: rimuove le originali ormai ridondanti (e identificativi).
    3. AgeHandler & OutlierCapper: gestisce i valori numerici.
    4. FrequencyEncoder: encoding specifico per geo_level_2 e 3.
    5. CategoricalEncoder: One-Hot per il resto.
    6. NumericalScaler (opzionale): scaling finale.
    """
    steps = [
        ('feature_engineering', FunctionTransformer(add_engineered_features)),
        ('cleaner', DataCleaner()),
        ('age_handler', AgeHandler()),
        ('outlier_capper', OutlierCapper()),
        ('geo_freq_encoder', FrequencyEncoder()),
        ('cat_encoder', CategoricalEncoder()),
    ]
    
    if scale_numeric:
        steps.append(('scaler', NumericalScaler()))
        
    return steps

def get_preprocessing_pipeline(scale_numeric=True):
    """
    Costruisce la pipeline di preprocessing completa.
    """
    return Pipeline(get_preprocessing_steps(scale_numeric))

def make_complete_pipeline(model, scale_numeric=True):
    """
    Crea una pipeline "piatta" (non nidificata) che include sia 
    preprocessing che modello.
    """
    steps = get_preprocessing_steps(scale_numeric)
    steps.append(('model', model))
    
    return Pipeline(steps)
