import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Classe per la gestione degli outlier tramite il metodo IQR (Interquartile Range).
    Modulo della pipeline modulare di preprocessing.
    """
    def __init__(self, columns=None):
        self.columns = columns or ["area_percentage", "height_percentage"]
        self.bounds_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X = X.copy()
        for col, bound in self.bounds_.items():
            if col in X.columns:
                X[col] = np.where(X[col] > bound, bound, X[col])
        return X
