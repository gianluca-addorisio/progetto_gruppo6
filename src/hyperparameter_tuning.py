"""
Ottimizzazione degli iperparametri dei modelli.

Il modulo definisce una classe di supporto per eseguire RandomizedSearchCV su
RandomForest, XGBoost, LightGBM e pipeline sklearn complete. Viene usato sia
per tuning diretto dei modelli base sia per il tuning delle pipeline durante
gli esperimenti.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelTuner:
    """
    Gestisce l'ottimizzazione degli iperparametri per i principali modelli.

    La classe centralizza le griglie di ricerca e usa StratifiedKFold per
    mantenere la distribuzione delle classi durante la cross-validation.
    """

    def __init__(self, random_state=42, n_splits=5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    def tune_random_forest(self, X, y, n_iter=20):
        """Ottimizza gli iperparametri di Random Forest."""
        rf = RandomForestClassifier(random_state=self.random_state)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        return self._run_search(rf, param_grid, X, y, n_iter)

    def tune_xgboost(self, X, y, n_iter=20):
        """Ottimizza gli iperparametri di XGBoost."""
        xgb = XGBClassifier(random_state=self.random_state, eval_metric="mlogloss")
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 10, 15],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10],
        }
        return self._run_search(xgb, param_grid, X, y, n_iter)

    def tune_lightgbm(self, X, y, n_iter=20):
        """Ottimizza gli iperparametri di LightGBM."""
        lgbm = LGBMClassifier(random_state=self.random_state, verbosity=-1)
        param_grid = {
            "n_estimators": [100, 300, 500],
            "num_leaves": [31, 63, 127, 255],
            "max_depth": [-1, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [20, 50, 100],
        }
        return self._run_search(lgbm, param_grid, X, y, n_iter)

    def _run_search(self, estimator, param_grid, X, y, n_iter):
        """Esegue la RandomizedSearchCV per uno stimatore e una griglia dati."""
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_micro",
            cv=self.cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_score_, search

    def get_param_grid(self, model_name):
        """Restituisce il grid di parametri per un dato modello."""
        if model_name == "RandomForest":
            return {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            }
        elif model_name == "XGBoost":
            return {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [3, 6, 10],
                "model__learning_rate": [0.01, 0.1],
            }
        elif model_name == "LightGBM":
            return {
                "model__n_estimators": [100, 300, 500],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.01, 0.1],
            }
        elif model_name == "StackingEnsemble":
            return {
                "model__final_estimator__C": [0.1, 1.0, 10.0],
                "model__final_estimator__solver": ["lbfgs", "liblinear"],
            }
        return {}

    def tune_pipeline(self, pipeline, model_name, X, y, n_iter=20):
        """
        Ottimizza una pipeline sklearn agendo solo sui parametri del modello.

        Questo metodo va usato quando la pipeline non contiene uno step
        di feature selection. I parametri devono quindi avere prefisso
        'model__'.
        """
        param_grid = self.get_param_grid(model_name)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_micro",
            cv=self.cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
        )

        search.fit(X, y)
        return search.best_estimator_, search.best_params_, search.best_score_