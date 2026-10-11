"""Train and evaluate the at-risk classifier, and save EDA charts."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                            precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_gen import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def metrics_for(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }


def eda_charts(df):
    static = ROOT / "static"
    static.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, g, c in [("Not at risk", df[df.at_risk == 0], "#2b8a6e"),
                        ("At risk", df[df.at_risk == 1], "#d1495b")]:
        ax.hist(g.attendance_pct, bins=18, alpha=0.75, label=label, color=c)
    ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Students")
    ax.set_title("Attendance distribution by outcome"); ax.legend()
    fig.tight_layout(); fig.savefig(static / "eda_attendance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(df.study_hours_week, df.prev_grade,
                c=df.at_risk.map({0: "#2b8a6e", 1: "#d1495b"}), s=14, alpha=0.7)
    ax.set_xlabel("Study hours / week"); ax.set_ylabel("Previous grade (0-20)")
    ax.set_title("Study hours vs previous grade (red = at risk)")
    fig.tight_layout(); fig.savefig(static / "eda_scatter.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    rate = df.groupby("past_failures").at_risk.mean() * 100
    ax.bar(rate.index.astype(str), rate.values, color="#3d5a80")
    ax.set_xlabel("Past failures"); ax.set_ylabel("% at risk")
    ax.set_title("At-risk rate by number of past failures")
    fig.tight_layout(); fig.savefig(static / "eda_failures.png", dpi=130); plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "data" / "students.csv")
    X, y = df[FEATURES], df["at_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    rf = RandomForestClassifier(n_estimators=250, max_depth=6, class_weight="balanced", random_state=7)
    lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)

    res = {"Logistic Regression": metrics_for(lr, X_te, y_te),
            "Random Forest": metrics_for(rf, X_te, y_te)}
    cv = cross_val_score(lr, X, y, cv=5, scoring="f1")
    coefs = lr.named_steps["logisticregression"].coef_[0]
    info = {
        "dataset": {"rows": int(len(df)), "at_risk_share": float(y.mean()),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                    "type": "synthetic"},
        "models": res,
        "deployed_model": "Logistic Regression",
        "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
        "coefficients": dict(zip(FEATURES, map(float, coefs))),
        "rf_importance": dict(zip(FEATURES, map(float, rf.feature_importances_))),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(lr, ROOT / "models" / "risk_model.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(info, indent=2))
    eda_charts(df)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    order = np.argsort(np.abs(coefs))
    ax.barh(np.array(FEATURES)[order], coefs[order],
            color=["#d1495b" if c > 0 else "#2b8a6e" for c in coefs[order]])
    ax.set_title("Logistic Regression coefficients (+ raises risk)")
    fig.tight_layout(); fig.savefig(ROOT / "static" / "eval_coefs.png", dpi=130); plt.close(fig)

    for name, m in res.items():
        print(f"{name:20s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")
    print(f"5-fold CV F1 (LR): {cv.mean():.3f} +- {cv.std():.3f}")


if __name__ == "__main__":
    main()
