import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Gestisce la trasformazione delle variabili categoriche in variabili numeriche (One-Hot Encoding).
    Recuperato da: src/preprocessing.py e op_3_handle_categorical_features.py
    """
    def __init__(self, categorical_cols=None):
        self.categorical_cols = categorical_cols
        self.encoder = None
        self.encoded_feature_names_ = None

    def fit(self, X, y=None):
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
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
            encoded_data = self.encoder.transform(X[self.categorical_cols])
            df_encoded = pd.DataFrame(
                encoded_data, 
                columns=self.encoded_feature_names_,
                index=X.index
            )
            X = X.drop(columns=self.categorical_cols)
            X = pd.concat([X, df_encoded], axis=1)
        return X
