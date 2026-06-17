import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AgeHandler(BaseEstimator, TransformerMixin):
    """
    Gestisce la logica specifica per la colonna 'age', inclusa la creazione di flag per edifici storici.
    Modulo della pipeline modulare di preprocessing.
    """
    def __init__(self, age_col='age'):
        self.age_col = age_col
        self.max_normal_age_ = None

    def fit(self, X, y=None):
        if self.age_col in X.columns:
            normal_ages = X.loc[X[self.age_col] < 995, self.age_col]
            if not normal_ages.empty:
                self.max_normal_age_ = normal_ages.max()
            else:
                self.max_normal_age_ = 200 
        return self

    def transform(self, X):
        X = X.copy()
        if self.age_col in X.columns:
            # Creazione flag 'is_historic' per age == 995
            X['is_historic'] = (X[self.age_col] == 995).astype(int)
            # Sostituzione di 995 con l'età massima normale
            if self.max_normal_age_ is not None:
                X.loc[X[self.age_col] == 995, self.age_col] = self.max_normal_age_
        return X
