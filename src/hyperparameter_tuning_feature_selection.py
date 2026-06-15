from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

class FeatureSelectionTuner:
    """
    Classe semplificata per l'ottimizzazione della Feature Selection.
    Usa il nome fisso 'feature_selector' per coerenza con la pipeline.
    """

    def __init__(self, random_state=42, n_splits=5):
        self.random_state = random_state
        self.n_splits = n_splits

    def tune_pipeline(self, pipeline, model_name, param_grid_model, X, y, n_iter=30):
        """
        Ottimizzazione congiunta di Feature Selection e Modello.
        Mantiene fisso il metodo di FS scelto dall'utente e ne ottimizza i parametri.
        """
        # Griglia per la Feature Selection (usando il nome fisso 'feature_selector')
        # Rimuoviamo 'fs_method' dalla griglia così rimane quello impostato nella pipeline
        param_grid = {
            'feature_selector__max_features_to_hold': randint(15, 45),
            'feature_selector__threshold': uniform(0, 0.05)
        }
        
        # Aggiungiamo i parametri del modello (che usano il prefisso 'model__')
        param_grid.update(param_grid_model)

        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_grid, 
            n_iter=n_iter, 
            cv=self.n_splits, 
            scoring='f1_micro',

            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X, y)
        
        return search.best_estimator_, search.best_params_, search.best_score_
