from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class KBestRFESelector(BaseEstimator, TransformerMixin):
    def __init__(
        self
        ,k=100
        ,n_features_to_select=30
        ,estimator=None
        ,score_func=f_classif
        ,random_state=20
    ):
        """
        Combined SelectKBest + RFE feature selector.

        Args:
            k (int): Number of top features to select using SelectKBest.
            n_features_to_select (int): Final number of features to retain via RFE.
            estimator (sklearn estimator): Estimator to use in RFE. Defaults to RandomForestClassifier.
            score_func (function): Scoring function for SelectKBest.
            random_state (int): Random state for reproducibility (used in default estimator).
        """
        self.k = k
        self.n_features_to_select = n_features_to_select
        self.estimator = estimator
        self.score_func = score_func
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = np.arange(X.shape[1])

        # SelectKBest
        self.kbest_ = SelectKBest(score_func=self.score_func, k=self.k)
        X_kbest = self.kbest_.fit_transform(X, y)
        self.kbest_features_ = self.feature_names_in_[self.kbest_.get_support()]

        # RFE
        if self.estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state)
        else:
            estimator = clone(self.estimator)

        self.rfe_ = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select)
        self.rfe_.fit(X_kbest, y)
        self.selected_features_ = self.kbest_features_[self.rfe_.get_support()]

        return self

    def transform(self, X):
        X_kbest = self.kbest_.transform(X)
        X_rfe = self.rfe_.transform(X_kbest)

        return pd.DataFrame(X_rfe, columns=self.selected_features_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_
