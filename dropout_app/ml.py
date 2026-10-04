"""Machine-learning part: train, predict, explain.

The model is trained on synthetic data the first time the app runs.
To use your own data, run:  python ml.py your_labeled_file.csv
(CSV needs the feature columns below plus a 'risk_label' column: Low / Medium / High)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")

FEATURES = ["attendance", "avg_marks", "backlogs", "fee_pending", "income_num", "entrance_score"]
INCOME_MAP = {"low": 0, "medium": 1, "high": 2}

# What a "healthy" student looks like. Used to work out which factors push risk up.
HEALTHY = {"attendance": 85, "avg_marks": 70, "backlogs": 0, "fee_pending": 0,
           "income_num": 1, "entrance_score": 65}


def make_synthetic_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    att = np.clip(rng.normal(75, 15, n), 30, 100)
    marks = np.clip(rng.normal(62, 15, n) + 0.2 * (att - 75), 15, 98)
    income = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    backlogs = rng.poisson(np.clip((60 - marks) / 12, 0, None))
    fee = (rng.random(n) < (0.12 + 0.2 * (income == 0))).astype(int)
    entrance = np.clip(rng.normal(60, 15, n) + 0.3 * (marks - 60), 10, 100)
    z = (0.045 * (75 - att) + 0.04 * (60 - marks) + 0.5 * backlogs + 1.0 * fee
         + 0.4 * (1 - income) + 0.02 * (60 - entrance) + rng.normal(0, 0.4, n))
    lo, hi = np.percentile(z, [55, 85])
    label = np.where(z <= lo, "Low", np.where(z <= hi, "Medium", "High"))
    return pd.DataFrame({"attendance": att, "avg_marks": marks, "backlogs": backlogs,
                         "fee_pending": fee, "income_num": income,
                         "entrance_score": entrance, "risk_label": label})


def train(df=None):
    if df is None:
        df = make_synthetic_data()
    X, y = df[FEATURES], df["risk_label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=1),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=1),
    }
    results, best_name, best_acc = {}, None, -1
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        results[name] = {
            "accuracy": round(acc, 3),
            "precision": round(precision_score(y_te, pred, average="macro", zero_division=0), 3),
            "recall": round(recall_score(y_te, pred, average="macro", zero_division=0), 3),
        }
        if acc > best_acc:
            best_name, best_acc = name, acc
    best = candidates[best_name]
    best.fit(X, y)  # final fit on all data
    joblib.dump(best, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"chosen": best_name, "results": results}, f, indent=2)
    return best_name, results


def _load():
    if not os.path.exists(MODEL_PATH):
        train()
    return joblib.load(MODEL_PATH)


_model = _load()


def reload_model():
    global _model
    _model = _load()


def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def _score(row_df):
    """Risk score from 0 to 1: P(High) + 0.5 * P(Medium)."""
    proba = _model.predict_proba(row_df)[0]
    p = dict(zip(_model.classes_, proba))
    return p.get("High", 0) + 0.5 * p.get("Medium", 0)


def _describe(feature, v):
    if feature == "attendance":
        return f"Low attendance ({v:.0f}%)"
    if feature == "avg_marks":
        return f"Low average marks ({v:.0f}%)"
    if feature == "backlogs":
        return f"{int(v)} backlog subject(s)"
    if feature == "fee_pending":
        return "Fee payment is pending"
    if feature == "income_num":
        return "Low family income bracket"
    return f"Low entrance score ({v:.0f})"


def predict(values):
    """values: dict with the FEATURES keys. Returns (level, score, reasons list)."""
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    level = str(_model.predict(row)[0])
    base = _score(row)
    healthy_row = pd.DataFrame([[HEALTHY[f] for f in FEATURES]], columns=FEATURES)
    impacts = []
    for f in FEATURES:
        # Drop: risk falls when only this factor is made healthy.
        alt = row.copy()
        alt[f] = HEALTHY[f]
        drop = base - _score(alt)
        # Rise: risk climbs when only this factor is made as bad as the student's.
        alone = healthy_row.copy()
        alone[f] = values[f]
        rise = _score(alone) - _score(healthy_row)
        impact = (drop + rise) / 2  # average of both views, so it works even when risk is maxed out
        if impact > 0.02:
            impacts.append((impact, _describe(f, values[f])))
    impacts.sort(reverse=True)
    reasons = [text for _, text in impacts[:3]]
    if not reasons:
        reasons = ["No major risk factors found"]
    return level, round(float(base), 3), reasons


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else None
    if data is not None and "income_level" in data.columns:
        data["income_num"] = data["income_level"].str.lower().map(INCOME_MAP)
    name, res = train(data)
    print("Chosen model:", name)
    print(json.dumps(res, indent=2))
