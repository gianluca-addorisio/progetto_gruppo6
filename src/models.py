"""
Definizione dei modelli utilizzati nel progetto.

Il modulo raccoglie i costruttori dei baseline model, dei modelli avanzati e
degli ensemble. Le funzioni restituiscono istanze sklearn compatibili con la
pipeline modulare e permettono di passare stimatori già tunati agli ensemble.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_dummy_classifier():
    """Restituisce il classificatore baseline basato sulla classe maggioritaria."""
    return DummyClassifier(strategy="most_frequent")


def get_logistic_regression():
    """Restituisce il classificatore baseline di regressione logistica."""
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def get_decision_tree():
    """Restituisce il classificatore baseline ad albero di decisione."""
    return DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    )


def get_random_forest_model():
    """Restituisce il modello Random Forest utilizzato nelle comparazioni avanzate."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgboost_model():
    """
    Restituisce il modello XGBoost utilizzato nelle comparazioni avanzate
    e negli ensemble.
    """
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )


def get_lightgbm_model():
    """
    Restituisce il modello LightGBM utilizzato nelle comparazioni avanzate.
    """
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def get_voting_ensemble(rf_model=None, xgb_model=None, lgbm_model=None):
    """
    Restituisce un ensemble di voto soft basato su Random Forest, XGBoost e LightGBM.

    Quando sono forniti stimatori base già ottimizzati, questi vengono usati
    direttamente all'interno dell'ensemble.
    """
    estimators = [
        ("rf", rf_model if rf_model is not None else get_random_forest_model()),
        ("xgb", xgb_model if xgb_model is not None else get_xgboost_model()),
        ("lgbm", lgbm_model if lgbm_model is not None else get_lightgbm_model()),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )


def get_stacking_ensemble(rf_model=None, xgb_model=None, lgbm_model=None):
    """
    Restituisce un ensemble di stacking basato su Random Forest, XGBoost e LightGBM.

    Quando sono forniti stimatori base già ottimizzati, questi vengono usati
    direttamente all'interno dell'ensemble.
    """
    estimators = [
        ("rf", rf_model if rf_model is not None else get_random_forest_model()),
        ("xgb", xgb_model if xgb_model is not None else get_xgboost_model()),
        ("lgbm", lgbm_model if lgbm_model is not None else get_lightgbm_model()),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
    )
