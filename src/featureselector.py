from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_selection import FeatureSelection


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for optional feature selection.

    The selector wraps the scoring methods implemented in FeatureSelection and
    keeps the best features according to a minimum score threshold and a maximum
    number of retained variables.
    """

    SUPPORTED_METHODS = {
        "rf",
        "xgb",
        "ctb",
        "corr_matrix",
        "chi2",
        "mu",
        "rlf",
        "rfe",
        "sfs",
    }

    def __init__(
        self,
        fs_method: str,
        threshold: float = 0.005,
        max_features_to_hold: int = 30,
    ):
        self.fs_method = fs_method
        self.threshold = threshold
        self.max_features_to_hold = max_features_to_hold
        self.selected_features_ = None
        self.scores_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Select the most relevant features according to the configured method."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        fs = FeatureSelection()
        scores = self._compute_scores(fs, X, y)

        scores = scores.dropna().sort_values(ascending=False)
        self.scores_ = scores

        selected_features = scores[scores >= self.threshold].index.tolist()

        if not selected_features:
            selected_features = scores.index.tolist()

        self.selected_features_ = selected_features[: self.max_features_to_hold]

        if not self.selected_features_:
            raise ValueError("FeatureSelector did not select any feature.")

        print(
            f"FeatureSelector ({self.fs_method}): selected "
            f"{len(self.selected_features_)} features out of {X.shape[1]}"
        )

        return self

    def transform(self, X: pd.DataFrame):
        """Reduce the dataset to the features selected during fit."""
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector must be fitted before transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        existing_features = [
            feature for feature in self.selected_features_ if feature in X.columns
        ]
        return X[existing_features]

    def get_feature_names_out(self, input_features=None):
        """Return the selected feature names for scikit-learn compatibility."""
        return self.selected_features_

    def _compute_scores(
        self,
        feature_selection: FeatureSelection,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Compute feature scores using the configured selection method."""
        if self.fs_method not in self.SUPPORTED_METHODS:
            allowed = ", ".join(sorted(self.SUPPORTED_METHODS))
            raise ValueError(
                f"Feature-selection method not recognized: {self.fs_method}. "
                f"Allowed values: {allowed}."
            )

        if self.fs_method == "rf":
            return feature_selection.random_forest_importances(X, y)
        if self.fs_method == "xgb":
            return feature_selection.xgboost_importances(X, y)
        if self.fs_method == "ctb":
            return feature_selection.catboost_importances(X, y)
        if self.fs_method == "corr_matrix":
            return feature_selection.correlation_ranking(X, y)
        if self.fs_method == "chi2":
            return feature_selection.chi_square_scores(X, y)
        if self.fs_method == "mu":
            return feature_selection.information_gain_scores(X, y)
        if self.fs_method == "rlf":
            return feature_selection.relief_importances(X, y)
        if self.fs_method == "rfe":
            return feature_selection.rfe_selection(
                X,
                y,
                n_features_to_select=self.max_features_to_hold,
            )
        if self.fs_method == "sfs":
            return feature_selection.sfs_selection(
                X,
                y,
                n_features_to_select=self.max_features_to_hold,
            )

        raise RuntimeError("Unreachable feature-selection branch.")
