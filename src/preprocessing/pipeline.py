from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Importiamo i nostri moduli personalizzati
from .cleaner import DataCleaner
from .age_handler import AgeHandler
from .outliers import OutlierCapper
from .encoding import CategoricalEncoder
from .scaling import NumericalScaler

# Importiamo la funzione per il feature engineering esistente
from ..features import add_engineered_features

def get_preprocessing_steps(scale_numeric=True):
    """
    Restituisce la lista dei passi di preprocessing.
    Invece di restituire una Pipeline nidificata, restituiamo i passi 
    per permettere di creare una pipeline "piatta", più robusta.
    """
    steps = [
        ('cleaner', DataCleaner()),
        ('age_handler', AgeHandler()),
        ('outlier_capper', OutlierCapper()),
        ('feature_engineering', FunctionTransformer(add_engineered_features)),
        ('encoder', CategoricalEncoder()),
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
    preprocessing che modello. Questo evita errori di 'NotFittedError'.
    """
    steps = get_preprocessing_steps(scale_numeric)
    steps.append(('model', model))
    
    return Pipeline(steps)
