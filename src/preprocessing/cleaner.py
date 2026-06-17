import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Esegue operazioni di pulizia e pruning (rimozione feature) sul dataset.
    
    Modulo della pipeline modulare di preprocessing.
    
    Operazioni:
    1. Rimozione di identificativi e target.
    2. Rimozione di feature ridondanti (sostituite da engineered features).
    3. Rimozione di feature a bassa informatività selezionate nella nuova strategia.
    4. Imputazione di sicurezza per valori nulli.
    """
    
    def __init__(self, additional_cols_to_drop=None):
        self.additional_cols_to_drop = additional_cols_to_drop

        # Feature da rimuovere nella pipeline finale.
        self.final_drop_features = [
            'building_id', 
            'damage_grade',
            'building_volume_proxy',
            "age_clipped",
            "age_group",
            "family_count_group",
            "floor_count_group",
            "plan_configuration",
            "legal_ownership_status",
        ]
        if self.additional_cols_to_drop:
            self.final_drop_features.extend(self.additional_cols_to_drop)

    def _get_dynamic_drop_features(self, df: pd.DataFrame) -> list[str]:
        """Identifica le feature binarie originali da rimuovere dopo l'aggregazione."""
        return [
            col for col in df.columns
            if col.startswith("has_superstructure_") or 
               col == "has_secondary_use" or 
               col.startswith("has_secondary_use_")
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Identificazione feature dinamiche (superstructure, secondary use)
        dynamic_drops = self._get_dynamic_drop_features(X)
        
        # 2. Unione con le feature fisse
        all_to_drop = list(set(self.final_drop_features + dynamic_drops))
        
        # 3. Rimozione effettiva
        existing_cols_to_drop = [col for col in all_to_drop if col in X.columns]
        if existing_cols_to_drop:
            X = X.drop(columns=existing_cols_to_drop)
            
        # 4. Gestione valori nulli (Fallback di sicurezza)
        if X.isnull().values.any():
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna('missing')
                    
        return X
