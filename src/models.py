from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .config import RANDOM_STATE

def get_dummy_classifier():
    """Returns a baseline DummyClassifier."""
    return DummyClassifier(strategy="most_frequent")

def get_logistic_regression():
    """Returns a configured LogisticRegression."""
    return LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE)

def get_decision_tree():
    """Returns a simple DecisionTreeClassifier."""
    return DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)

def get_random_forest_model():
    """Returns a configured RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def get_xgboost_model():
    """Returns a configured XGBClassifier."""
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )

def get_lightgbm_model():
    """Returns a configured LGBMClassifier."""
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

