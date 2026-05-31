import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Normalizza le variabili numeriche in modo che abbiano media 0 e deviazione standard 1.
    
    RECUPERATO DA: src/preprocessing.py e op_4_scale_numerical_features.py
    
    GIUSTIFICAZIONE:
    Molti modelli (come quelli lineari o basati su distanze) performano meglio se le 
    variabili numeriche sono sulla stessa scala. 
    Esempio: 'age' (0-200) e 'area_percentage' (0-100) devono essere confrontabili.
    """
    
    def __init__(self, numeric_cols=None):
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Identifica le colonne numeriche e calcola media e deviazione standard.
        """
        if self.numeric_cols is None:
            # Selezioniamo automaticamente le colonne numeriche (int e float)
            # Escludiamo quelle che sembrano essere binarie (0 e 1) per non rovinarle
            all_numeric = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
            self.numeric_cols = [
                col for col in all_numeric 
                if not X[col].isin([0, 1]).all() # Esclude le flag binarie (has_superstructure_...)
            ]
            
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
            
        return self

    def transform(self, X):
        """
        Applica la normalizzazione calcolata nel fit.
        """
        X = X.copy()
        
        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
            
        return X
