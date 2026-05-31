import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Esegue operazioni di pulizia generale sul dataset.
    
    RECUPERATO DA: src/preprocessing.py (prepare_features) e logiche sparse in op_*.py
    
    GIUSTIFICAZIONE:
    Centralizziamo le operazioni di "base" che devono avvenire prima di qualsiasi trasformazione:
    1. Rimozione di colonne inutili come 'building_id' (che è un identificativo, non una feature).
    2. Rimozione accidentale del target se presente nel DataFrame delle feature.
    3. Gestione di eventuali valori nulli (Simple Imputation come fallback di sicurezza).
    """
    
    def __init__(self, cols_to_drop=['building_id', 'damage_grade']):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        # Il cleaner non ha bisogno di "imparare" nulla dai dati per ora,
        # ma manteniamo la struttura per compatibilità.
        return self

    def transform(self, X):
        """
        Esegue la pulizia effettiva.
        """
        X = X.copy()
        
        # 1. Rimozione colonne non necessarie
        existing_cols_to_drop = [col for col in self.cols_to_drop if col in X.columns]
        if existing_cols_to_drop:
            X = X.drop(columns=existing_cols_to_drop)
            
        # 2. Gestione valori nulli (Fallback di sicurezza)
        # Sebbene il dataset Richter sia noto per non avere nulli, 
        # è buona norma avere un'azione di difesa se si usa il codice su nuovi dati.
        if X.isnull().values.any():
            # Riempiamo i nulli numerici con la mediana e quelli categorici con 'missing'
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna('missing')
                    
        return X
