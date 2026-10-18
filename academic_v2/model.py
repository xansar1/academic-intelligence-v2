"""ML selection/evaluation with a truly separate test subset (synthetic-only)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES


@dataclass
class ModelBundle:
    pipeline: object
    scores: dict
    reference: dict
    population_positive_rate: float
    test_size: int


def train_demo_model(df: pd.DataFrame, seed: int = 42) -> ModelBundle:
    X = df[FEATURES].copy()
    y = df["Needs Support Next Term"].astype(int)
    if len(X) < 100 or y.nunique() != 2:
        raise ValueError("Model requires at least 100 records and both classes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    threshold = .30  # predeclared demo threshold; not tuned on test observations
    flag = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, flag, labels=[0, 1])
    stats = {
        "pr_auc": round(float(average_precision_score(y_test, prob)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 3),
        "precision": round(float(precision_score(y_test, flag, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, flag, zero_division=0)), 3),
        "brier": round(float(brier_score_loss(y_test, prob)), 3),
        "baseline_pr_auc": round(float(y_test.mean()), 3),
        "threshold": threshold,
        "true_negative": int(cm[0, 0]), "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]), "true_positive": int(cm[1, 1]),
    }
    return ModelBundle(pipe, stats, X_train.median().to_dict(), float(y.mean()), len(X_test))


def score_record(bundle: ModelBundle, record: dict) -> dict:
    values = {}
    for key in FEATURES:
        try:
            num = float(record[key])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid input: {key}") from exc
        if not np.isfinite(num) or not 0 <= num <= 100:
            raise ValueError(f"{key} must be between 0 and 100")
        values[key] = num
    frame = pd.DataFrame([values], columns=FEATURES)
    probability = float(bundle.pipeline.predict_proba(frame)[0, 1])
    score = float(bundle.pipeline.decision_function(frame)[0])
    drivers=[]
    for key in FEATURES:
        baseline=values.copy()
        baseline[key]=bundle.reference[key]
        baseline_score=float(bundle.pipeline.decision_function(pd.DataFrame([baseline],columns=FEATURES))[0])
        drivers.append({"feature":key,"log_odds_shift":round(score-baseline_score,3)})
    return {"probability": probability, "review_suggested": probability >= bundle.scores["threshold"],
            "contributions": sorted(drivers,key=lambda item:abs(item["log_odds_shift"]),reverse=True)}
