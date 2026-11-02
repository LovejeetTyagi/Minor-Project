"""Heart Disease Prediction System - Flask application.

Educational project only. Not a medical diagnostic tool.
"""
import json
import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, url_for)

import database
from insights import explain_prediction, health_tips
from report import build_pdf

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DISCLAIMER = (
    "This system is for educational purposes only. It is not a medical "
    "diagnostic tool. Please consult a qualified doctor for medical advice."
)

# ---------------------------------------------------------------------------
# 2. Load trained model and scaler
# ---------------------------------------------------------------------------
def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run notebooks/model_training.ipynb first "
                "to train the model and create the files in models/."
            )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


model, scaler = load_artifacts()
CLASS_INDEX = list(model.classes_).index(1)
app.logger.info("Model (%s) and scaler loaded.", type(model).__name__)


def load_json(path):
    """Optional files written by the notebook. Returns None if missing/invalid."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


METRICS = load_json(BASE_DIR / "models" / "metrics.json")      # for the Models page
REFERENCE = load_json(BASE_DIR / "models" / "reference.json")  # typical values for explanations
if METRICS is None or REFERENCE is None:
    app.logger.warning("metrics.json / reference.json not found - re-run the notebook "
                       "to enable the Models page and 'Why this result?'.")

try:
    database.init_db()
except Exception:  # history is optional; never stop the app because of it
    app.logger.exception("Could not initialise the history database")

# ---------------------------------------------------------------------------
# 3. Feature definitions (single source of truth for the form AND validation)
#    The order here MUST match the column order used in training.
# ---------------------------------------------------------------------------
FIELDS = [
    {"name": "age", "label": "Age", "type": "number", "min": 1, "max": 120, "step": 1,
     "unit": "years", "placeholder": "e.g. 54"},
    {"name": "sex", "label": "Sex", "type": "select",
     "options": [(1, "Male"), (0, "Female")]},
    {"name": "cp", "label": "Chest pain type", "type": "select",
     "options": [(0, "Typical angina"), (1, "Atypical angina"),
                 (2, "Non-anginal pain"), (3, "Asymptomatic")]},
    {"name": "trestbps", "label": "Resting blood pressure", "type": "number",
     "min": 50, "max": 300, "step": 1, "unit": "mm Hg", "placeholder": "e.g. 130"},
    {"name": "chol", "label": "Serum cholesterol", "type": "number",
     "min": 50, "max": 700, "step": 1, "unit": "mg/dl", "placeholder": "e.g. 240"},
    {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "type": "select",
     "options": [(0, "No"), (1, "Yes")]},
    {"name": "restecg", "label": "Resting ECG result", "type": "select",
     "options": [(0, "Normal"), (1, "ST-T wave abnormality"),
                 (2, "Left ventricular hypertrophy")]},
    {"name": "thalach", "label": "Maximum heart rate achieved", "type": "number",
     "min": 40, "max": 250, "step": 1, "unit": "bpm", "placeholder": "e.g. 150"},
    {"name": "exang", "label": "Exercise-induced angina", "type": "select",
     "options": [(0, "No"), (1, "Yes")]},
    {"name": "oldpeak", "label": "ST depression (oldpeak)", "type": "number",
     "min": 0, "max": 10, "step": 0.1, "unit": "", "placeholder": "e.g. 1.0"},
    {"name": "slope", "label": "Slope of peak exercise ST segment", "type": "select",
     "options": [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")]},
    {"name": "ca", "label": "Major vessels coloured by fluoroscopy", "type": "select",
     "options": [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4")]},
    {"name": "thal", "label": "Thalassemia", "type": "select",
     "options": [(1, "Fixed defect"), (2, "Normal"), (3, "Reversible defect"),
                 (0, "Not recorded")]},
]
FEATURES = [f["name"] for f in FIELDS]


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------
def validate_input(data):
    """Return (values, errors). `errors` maps field name -> message."""
    if not isinstance(data, dict):
        return {}, {"_form": "Input must be a JSON object."}

    values, errors = {}, {}
    for field in FIELDS:
        name = field["name"]
        raw = data.get(name)

        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors[name] = "This field is required."
            continue
        if isinstance(raw, bool):
            errors[name] = "Must be a number."
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Must be a number."
            continue
        if not math.isfinite(number):
            errors[name] = "Must be a finite number."
            continue

        if field["type"] == "select":
            allowed = [value for value, _ in field["options"]]
            if not number.is_integer() or int(number) not in allowed:
                errors[name] = f"Must be one of {allowed}."
                continue
            number = int(number)
        elif not field["min"] <= number <= field["max"]:
            errors[name] = f"Must be between {field['min']} and {field['max']}."
            continue

        values[name] = number
    return values, errors


# ---------------------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------------------
def risk_level(probability):
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Moderate"
    return "High"


def run_prediction(values):
    frame = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(frame)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][CLASS_INDEX])
    return {
        "prediction": prediction,
        "label": ("Heart disease likely (model output)" if prediction == 1
                  else "Heart disease unlikely (model output)"),
        "probability": round(probability, 4),
        "percent": round(probability * 100, 1),
        "risk": risk_level(probability),
    }


def proba_fn(frame):
    """Probability of heart disease for every row of a DataFrame."""
    return model.predict_proba(scaler.transform(frame))[:, CLASS_INDEX]


FIELD_BY_NAME = {f["name"]: f for f in FIELDS}


def display_value(field, value):
    if field["type"] == "select":
        return dict(field["options"])[value]
    return f"{value:g} {field['unit']}".strip()


def summarise(values):
    return [(f["label"], display_value(f, values[f["name"]])) for f in FIELDS]


def analyse(values):
    """Prediction + 'why' factors + health tips for one validated input."""
    result = run_prediction(values)
    factors = []
    if REFERENCE:
        for item in explain_prediction(values, REFERENCE, FEATURES, proba_fn):
            field = FIELD_BY_NAME[item["name"]]
            factors.append({"name": item["name"], "label": field["label"],
                            "value": display_value(field, values[item["name"]]),
                            "effect": round(item["effect"], 1)})
    return result, factors, health_tips(values, result["risk"])


def save_history(source, values, result):
    try:
        database.save_prediction(source, values, result)
    except Exception:
        app.logger.exception("Could not save prediction history")


# ---------------------------------------------------------------------------
# 6. Web pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html", model_name=type(model).__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "GET":
        return render_template("predict.html", fields=FIELDS, form={}, errors={})

    form = request.form.to_dict()
    values, errors = validate_input(form)
    if errors:
        return render_template("predict.html", fields=FIELDS, form=form,
                               errors=errors), 400

    try:
        result, factors, tips = analyse(values)
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("error.html", code=500,
                               message="Prediction failed. Please try again."), 500

    save_history("web", values, result)
    factor_max = max((abs(f["effect"]) for f in factors), default=1) or 1
    return render_template("result.html", result=result, values=values,
                           summary=summarise(values), factors=factors,
                           factor_max=factor_max, tips=tips, disclaimer=DISCLAIMER)


@app.route("/report", methods=["POST"])
def report():
    values, errors = validate_input(request.form.to_dict())
    if errors:
        return render_template("error.html", code=400,
                               message="Invalid data for the report."), 400
    try:
        result, factors, tips = analyse(values)
        pdf = build_pdf(result, summarise(values), factors, tips, DISCLAIMER)
    except Exception:
        app.logger.exception("Report generation failed")
        return render_template("error.html", code=500,
                               message="Could not create the PDF report."), 500
    return send_file(pdf, mimetype="application/pdf", as_attachment=True,
                     download_name="heart-risk-report.pdf")


@app.route("/models")
def models_page():
    return render_template("models.html", metrics=METRICS)


@app.route("/history")
def history_page():
    try:
        rows = database.get_history()
        stats = database.get_stats()
    except Exception:
        app.logger.exception("Could not read history")
        rows, stats = [], None
    for row in rows:
        inputs = row["inputs"]
        row["summary"] = (f"Age {inputs.get('age'):g}, "
                          f"{'Male' if inputs.get('sex') == 1 else 'Female'}")
    return render_template("history.html", rows=rows, stats=stats)


@app.route("/history/clear", methods=["POST"])
def history_clear():
    try:
        database.clear_history()
    except Exception:
        app.logger.exception("Could not clear history")
    return redirect(url_for("history_page"))


# ---------------------------------------------------------------------------
# 7. JSON API
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route("/api/info")
def api_info():
    return jsonify({"model": type(model).__name__, "features": FEATURES,
                    "total_features": len(FEATURES), "status": "ready"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"success": False, "error": "Please send valid JSON."}), 400

    values, errors = validate_input(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        result, factors, _tips = analyse(values)
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction failed."}), 500

    save_history("api", values, result)
    return jsonify({
        "success": True,
        "prediction": result["prediction"],
        "prediction_label": result["label"],
        "model_probability_score": result["probability"],
        "risk_level": result["risk"],
        "top_factors": [{"feature": f["name"], "effect_points": f["effect"]} for f in factors],
        "disclaimer": DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# 8. Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Requested resource not found."}), 404
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "HTTP method not allowed."}), 405
    return render_template("error.html", code=405, message="Method not allowed."), 405


# ---------------------------------------------------------------------------
# 9. Run (development only; production uses gunicorn via Procfile)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1")
