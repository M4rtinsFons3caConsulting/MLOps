"""
This is a pipeline 'model_assessment'
generated using Kedro 0.19.14
"""

import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import timedelta
from .purged_kfold import PurgedKFold
from .feature_selection import KBestRFESelector
from .scaler import FeatureScaler


def objective(trial, X, y):
    # Hyperparameters to optimize
    k = trial.suggest_int('k', 5, 20)
    C = trial.suggest_loguniform('C', 1e-4, 1e2)

    # Create pipeline
    pipeline = Pipeline([
        ('feature_selection', KBestRFESelector(score_func=f_classif, k=k)),
        ('scaler', FeatureScaler()),
        ('model', LogisticRegression(C=C, max_iter=1000))
    ])

    # Initialize custom CV
    cv = PurgedKFold(n_splits=5, purging_window=timedelta(days=1))

    scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit and score manually
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        score = accuracy_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)

def run_optuna_optimization(X_train, y_train, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    return study.best_params, study.best_value