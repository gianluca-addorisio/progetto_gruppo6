from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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