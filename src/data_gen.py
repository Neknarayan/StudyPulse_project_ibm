"""Generate a SYNTHETIC student-record dataset for the StudyPulse prototype.

The data is artificial (random, with plausible relationships between
features and outcome). It exists so the project runs offline. For a real
evaluation, replace data/students.csv with a real dataset that has the same
columns (for example the UCI "Student Performance" dataset, re-mapped) and
re-run `python -m src.train`.
"""
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = [
    "study_hours_week", "attendance_pct", "prev_grade", "absences",
    "past_failures", "assignment_pct", "sleep_hours",
    "parental_support", "internet_access",
]


def generate(n=600, seed=42):
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(8, 4, n), 0, 25)
    attendance = np.clip(rng.normal(82, 12, n), 35, 100)
    prev = np.clip(rng.normal(11.5, 3.2, n) + 0.08 * (study - 8), 0, 20)
    absences = np.clip(np.round((100 - attendance) / 4 + rng.poisson(1, n)), 0, 40)
    failures = rng.choice([0, 1, 2, 3], n, p=[0.72, 0.18, 0.07, 0.03])
    assign = np.clip(rng.normal(75, 18, n) + 0.3 * (attendance - 82), 0, 100)
    sleep = np.clip(rng.normal(6.8, 1.1, n), 3.5, 9.5)
    support = rng.integers(1, 6, n)
    internet = rng.choice([0, 1], n, p=[0.15, 0.85])

    z = lambda x: (x - x.mean()) / x.std()
    latent = (0.50 * z(prev) + 0.35 * z(attendance) + 0.30 * z(study)
                + 0.30 * z(assign) - 0.45 * failures - 0.12 * z(absences)
                + 0.10 * z(support) + 0.08 * z(sleep) + 0.15 * internet
                + rng.normal(0, 0.6, n))
    p_risk = 1 / (1 + np.exp(1.7 * (latent + 0.55)))
    at_risk = (rng.random(n) < p_risk).astype(int)

    return pd.DataFrame({
        "student_id": [f"S{i:03d}" for i in range(1, n + 1)],
        "study_hours_week": study.round(1),
        "attendance_pct": attendance.round(1),
        "prev_grade": prev.round(1),
        "absences": absences.astype(int),
        "past_failures": failures,
        "assignment_pct": assign.round(1),
        "sleep_hours": sleep.round(1),
        "parental_support": support,
        "internet_access": internet,
        "at_risk": at_risk,
    })


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "students.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}  (at-risk share: {df.at_risk.mean():.1%})")

