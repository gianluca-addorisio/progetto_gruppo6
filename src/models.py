from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .config import RANDOM_STATE

def get_random_forest_model():
    """Returns a configured RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def get_xgboost_model():
    """Returns a configured XGBClassifier."""
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

def get_lightgbm_model():
    """Returns a configured LGBMClassifier."""
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        random_state=RANDOM_STATE,
        verbosity=-1
    )
