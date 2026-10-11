"""Risk scoring, per-student explanation and rule-based support suggestions."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.data_gen import FEATURES

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "study_hours_week": "Study hours per week",
    "attendance_pct": "Attendance",
    "prev_grade": "Previous grade",
    "absences": "Absences",
    "past_failures": "Past failures",
    "assignment_pct": "Assignment completion",
    "sleep_hours": "Sleep hours",
    "parental_support": "Parental support",
    "internet_access": "Internet access",
}

ADVICE = {
    "study_hours_week": "Agree a weekly study timetable and check in on it every two weeks.",
    "attendance_pct": "Contact the student about attendance barriers and arrange catch-up sessions.",
    "prev_grade": "Offer foundation revision or peer tutoring for earlier week subjects.",
    "absences": "Review the reasons for absence and share missed material.",
    "past_failures": "Assign a mentor and a focused plan for the repeated subject.",
    "assignment_pct": "Break assignments into smaller milestones with reminders.",
    "sleep_hours": "Share sleep and wellbeing guidance; refer to student support if needed.",
    "parental_support": "Invite parent/guardian to a progress meeting.",
    "internet_access": "Provide campus lab slots or offline study material.",
}


def load():
    model = joblib.load(ROOT / "models" / "risk_model.joblib")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return model, metrics


def level(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"


def score(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]


def explain(model, row, top=3):
    """Top features pushing THIS student's risk up (coef x standardized value)."""
    scaler = model.named_steps["standardscaler"]
    coef = model.named_steps["logisticregression"].coef_[0]
    x = pd.DataFrame([row[FEATURES].astype(float)], columns=FEATURES)
    contrib = scaler.transform(x)[0] * coef
    out = []
    for i in np.argsort(contrib)[::-1][:top]:
        if contrib[i] > 0.05:
            f = FEATURES[i]
            out.append({"feature": f, "label": LABELS[f], "value": float(x.iloc[0, i]),
                       "impact": float(contrib[i]), "advice": ADVICE[f]})
    return out
