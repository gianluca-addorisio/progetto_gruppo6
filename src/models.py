from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_dummy_classifier():
    """Return the majority-class baseline classifier."""
    return DummyClassifier(strategy="most_frequent")


def get_logistic_regression():
    """Return the logistic-regression baseline classifier."""
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def get_decision_tree():
    """Return the decision-tree baseline classifier."""
    return DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    )


def get_random_forest_model():
    """Return the Random Forest model used in advanced comparisons."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgboost_model():
    """Return the XGBoost model used in advanced comparisons and ensembles."""
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )


def get_lightgbm_model():
    """Return the LightGBM model used in advanced comparisons."""
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
    Return a soft-voting ensemble based on Random Forest, XGBoost and LightGBM.
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
    Return a stacking ensemble based on Random Forest, XGBoost and LightGBM.
    
    When tuned base estimators are provided, they are used directly inside the ensemble.
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
