import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoder for high-cardinality categorical identifiers.
    Modulo della pipeline modulare di preprocessing.
    
    Durante il fit, calcola la frequenza relativa di ogni categoria.
    Durante il transform, mappa le categorie alle loro frequenze. 
    Le categorie non viste nel training vengono mappate a 0.0.
    """

    def __init__(self, geo_cols=['geo_level_2_id', 'geo_level_3_id']):
        self.geo_cols = geo_cols
        self.frequency_maps_: dict[str, pd.Series] = {}
        self.input_features_: list[str] = []

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.input_features_ = [col for col in self.geo_cols if col in X_df.columns]
        self.frequency_maps_ = {
            col: X_df[col].value_counts(normalize=True)
            for col in self.input_features_
        }
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        X_out = X_df.copy()

        for col in self.input_features_:
            freq_map = self.frequency_maps_[col]
            X_out[f"{col}_freq"] = (
                X_df[col]
                .map(freq_map)
                .fillna(0.0)
                .astype(float)
            )
            # Rimuoviamo l'originale per evitare ridondanza
            X_out = X_out.drop(columns=[col])

        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_freq" for col in self.input_features_], dtype=object)

    @staticmethod
    def _to_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Gestisce la trasformazione delle variabili categoriche in variabili numeriche (One-Hot Encoding).
    Esclude automaticamente le colonne destinate al Frequency Encoding.
    """
    def __init__(self, categorical_cols=None, exclude_cols=['geo_level_2_id', 'geo_level_3_id']):
        self.categorical_cols = categorical_cols
        self.exclude_cols = exclude_cols
        self.encoder = None
        self.encoded_feature_names_ = None

    def fit(self, X, y=None):
        if self.categorical_cols is None:
            # Prendiamo le colonne 'object', 'category' e 'string' per evitare warning futuri
            all_cat = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            # Aggiungiamo geo_level_1_id se presente e non già incluso
            if 'geo_level_1_id' in X.columns and 'geo_level_1_id' not in all_cat:
                all_cat.append('geo_level_1_id')
                
            self.categorical_cols = [c for c in all_cat if c not in self.exclude_cols]
        
        if self.categorical_cols:
            self.encoder = OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False,
                dtype='int'
            )
            self.encoder.fit(X[self.categorical_cols])
            self.encoded_feature_names_ = self.encoder.get_feature_names_out(self.categorical_cols)
        return self

    def transform(self, X):
        X = X.copy()
        if self.encoder and self.categorical_cols:
            existing_cols = [c for c in self.categorical_cols if c in X.columns]
            if not existing_cols:
                return X
                
            encoded_data = self.encoder.transform(X[existing_cols])
            df_encoded = pd.DataFrame(
                encoded_data, 
                columns=self.encoded_feature_names_,
                index=X.index
            )
            X = X.drop(columns=existing_cols)
            X = pd.concat([X, df_encoded], axis=1)
        return X
