"""StudyPulse - Student Academic Risk Early-Warning System (Flask prototype)."""
import pandas as pd
from flask import Flask, abort, render_template, request

from src import risk
from src.data_gen import FEATURES

app = Flask(__name__)
model, METRICS = risk.load()
DF = pd.read_csv("data/students.csv")
DF["risk_p"] = risk.score(model, DF)
DF["risk_level"] = DF.risk_p.map(risk.level)

DEFAULTS = {
    "study_hours_week": 8,
    "attendance_pct": 80,
    "prev_grade": 11,
    "absences": 4,
    "past_failures": 0,
    "assignment_pct": 75,
    "sleep_hours": 7,
    "parental_support": 3,
    "internet_access": 1,
}


@app.context_processor
def inject():
    return {"labels": risk.LABELS}


@app.route("/")
def dashboard():
    counts = (
        DF.risk_level.value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .astype(int)
    )
    top = DF.sort_values("risk_p", ascending=False).head(10)
    return render_template(
        "dashboard.html",
        total=len(DF),
        counts=counts.to_dict(),
        avg_att=DF.attendance_pct.mean(),
        avg_grade=DF.prev_grade.mean(),
        top=top.to_dict("records"),
        page="dashboard",
    )


@app.route("/data")
def data():
    return render_template(
        "data.html",
        rows=DF.head(15).to_dict("records"),
        n=len(DF),
        cols=list(DF.columns[:-2]),
        share=DF.at_risk.mean(),
        page="data",
    )


@app.route("/eda")
def eda():
    summary = (
        DF[FEATURES].describe().round(1).T.reset_index().to_dict("records")
    )
    return render_template("eda.html", summary=summary, page="eda")


@app.route("/model")
def model_page():
    return render_template("model.html", m=METRICS, page="model")


@app.route("/student/<sid>")
def student(sid):
    rows = DF[DF.student_id == sid]
    if rows.empty:
        abort(404)
    row = rows.iloc[0]
    return render_template(
        "student.html",
        s=row,
        factors=risk.explain(model, row),
        page="dashboard",
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    vals, result = dict(DEFAULTS), None
    if request.method == "POST":
        try:
            vals = {k: float(request.form[k]) for k in FEATURES}
        except (KeyError, ValueError):
            return "Invalid input", 400
        row = pd.Series(vals)
        p = float(model.predict_proba(pd.DataFrame([row[FEATURES]]))[0, 1])
        result = {"p": p, "level": risk.level(p), "factors": risk.explain(model, row)}
    return render_template("predict.html", vals=vals, result=result, page="predict")


if __name__ == "__main__":
    app.run(debug=True)
