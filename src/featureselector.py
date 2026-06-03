from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from src.feature_selection import FeatureSelection

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer per la selezione automatica delle feature basato sui metodi di FeatureSelection.
    Compatibile con le Pipeline di Scikit-Learn.
    """

    def __init__(self, fs_method: str, threshold=0.005, max_features_to_hold=30):
        """
        Parametri:
        - fs_method: metodo di selezione ('rf', 'xgb', 'ctb', 'corr_matrix', 'chi2', 'mu', 'rlf')
        - threshold: importanza minima per mantenere una feature
        - max_features_to_hold: numero massimo di feature da mantenere
        """
        self.fs_method = fs_method
        self.threshold = threshold
        self.max_features_to_hold = max_features_to_hold
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Apprende quali sono le feature migliori basandosi sul metodo scelto.
        """
        # Assicuriamoci che X sia un DataFrame per gestire i nomi delle colonne
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        fs = FeatureSelection()
        
        # 1. Calcolo degli score in base al metodo scelto
        if self.fs_method == 'rf':
            scores = fs.random_forest_importances(X, y)
        elif self.fs_method == 'xgb':
            scores = fs.xgboost_importances(X, y)
        elif self.fs_method == 'ctb':
            scores = fs.catboost_importances(X, y)
        elif self.fs_method == 'corr_matrix':
            scores = fs.correlation_ranking(X, y)
        elif self.fs_method == 'chi2':
            scores = fs.chi_square_scores(X, y)
        elif self.fs_method == 'mu':
            scores = fs.information_gain_scores(X, y)
        elif self.fs_method == 'rlf':
            scores = fs.relief_importances(X, y)
        else:
            raise ValueError(f"Metodo {self.fs_method} non riconosciuto.")

        # 2. Selezione delle feature che superano la soglia
        top_features = scores[scores >= self.threshold].index.tolist()
        
        # 3. Applicazione del limite massimo di feature
        self.selected_features_ = top_features[:self.max_features_to_hold]
        
        print(f"FeatureSelector ({self.fs_method}): selezionate {len(self.selected_features_)} feature su {X.shape[1]}")
        
        return self

    def transform(self, X: pd.DataFrame):
        """
        Riduce il dataset alle sole feature selezionate durante il fit.
        """
        if self.selected_features_ is None:
            raise ValueError("Il selettore deve essere addestrato (fit) prima di trasformare i dati.")
            
        # Conversione in DataFrame se necessario
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Restituiamo solo le feature selezionate presenti in X
        existing_features = [f for f in self.selected_features_ if f in X.columns]
        return X[existing_features]

    def get_feature_names_out(self, input_features=None):
        """Metodo di utilità per pipeline di Scikit-Learn."""
        return self.selected_features_
