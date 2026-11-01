"""Heart Disease Prediction System - Flask application.

Educational project only. Not a medical diagnostic tool.
"""
import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

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
app.logger.info("Model (%s) and scaler loaded.", type(model).__name__)

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
    class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(scaled)[0][class_index])
    return {
        "prediction": prediction,
        "label": ("Heart disease likely (model output)" if prediction == 1
                  else "Heart disease unlikely (model output)"),
        "probability": round(probability, 4),
        "percent": round(probability * 100, 1),
        "risk": risk_level(probability),
    }


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
        result = run_prediction(values)
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("error.html", code=500,
                               message="Prediction failed. Please try again."), 500

    summary = [(f["label"], dict(f["options"])[values[f["name"]]]
                if f["type"] == "select"
                else f"{values[f['name']]:g} {f['unit']}".strip())
               for f in FIELDS]
    return render_template("result.html", result=result, summary=summary,
                           disclaimer=DISCLAIMER)


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
        result = run_prediction(values)
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction failed."}), 500

    return jsonify({
        "success": True,
        "prediction": result["prediction"],
        "prediction_label": result["label"],
        "model_probability_score": result["probability"],
        "risk_level": result["risk"],
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
