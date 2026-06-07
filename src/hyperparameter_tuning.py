from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def tune_random_forest(
    X,
    y,
    n_iter=20,
    random_state=42
):
    rf = RandomForestClassifier(random_state=random_state)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_score_, search

def tune_xgboost(
    X,
    y,
    n_iter=20,
    random_state=42
):
    xgb = XGBClassifier(random_state=random_state, eval_metric="mlogloss")

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 6, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_score_, search

def tune_lightgbm(
    X,
    y,
    n_iter=20,
    random_state=42
):
    lgbm = LGBMClassifier(random_state=random_state, verbosity=-1)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "num_leaves": [31, 63, 127, 255],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [20, 50, 100],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_score_, search