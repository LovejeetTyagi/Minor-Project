"""Explanations and health tips shown next to a prediction."""
import pandas as pd


def explain_prediction(values, baseline, features, proba_fn, top=6, min_effect=0.5):
    """Model-agnostic 'why this result' using a simple what-if method.

    For each feature we replace the patient's value with a typical value (training
    median / most common category) and see how the risk score changes. A positive
    effect means that feature's value pushed the risk UP compared with a typical patient.
    Returns a list of {"name", "effect"} (effect in percentage points), biggest first.
    """
    rows = [dict(values)]
    for name in features:
        altered = dict(values)
        altered[name] = baseline[name]
        rows.append(altered)

    probabilities = proba_fn(pd.DataFrame(rows, columns=features))
    full = float(probabilities[0])

    effects = [
        {"name": name, "effect": (full - float(p)) * 100}
        for name, p in zip(features, probabilities[1:])
    ]
    effects = [e for e in effects if abs(e["effect"]) >= min_effect]
    effects.sort(key=lambda e: abs(e["effect"]), reverse=True)
    return effects[:top]


def health_tips(values, risk):
    """General, rule-based lifestyle guidance. Not medical advice."""
    personal = []

    if risk == "High":
        personal.append("Your result is in the High range. Please book an appointment with a "
                        "doctor or cardiologist for a proper check-up soon.")
    elif risk == "Moderate":
        personal.append("Your result is in the Moderate range. A routine check-up with a "
                        "doctor is a good idea.")
    else:
        personal.append("Your result is in the Low range. Keep up healthy habits and continue "
                        "regular check-ups.")

    bp = values["trestbps"]
    if bp >= 140:
        personal.append(f"Your resting blood pressure ({bp:g} mm Hg) is high. Reduce salt, "
                        "stay active, and get it measured regularly by a doctor.")
    elif bp >= 130:
        personal.append(f"Your resting blood pressure ({bp:g} mm Hg) is slightly raised. "
                        "Cut down on salt and monitor it regularly.")

    chol = values["chol"]
    if chol >= 240:
        personal.append(f"Your cholesterol ({chol:g} mg/dl) is high. Limit fried food and "
                        "saturated fat, eat more fibre, and discuss it with a doctor.")
    elif chol >= 200:
        personal.append(f"Your cholesterol ({chol:g} mg/dl) is borderline high. Eat more "
                        "vegetables, fruit and whole grains, and less fried food.")

    if values["fbs"] == 1:
        personal.append("Your fasting blood sugar is above 120 mg/dl. Limit sugary food and "
                        "drinks and get your blood sugar tested.")

    if values["exang"] == 1:
        personal.append("Chest pain during exercise should be checked by a doctor. Avoid "
                        "heavy exertion until you have been evaluated.")

    if values["oldpeak"] >= 2:
        personal.append("A high ST depression value is worth discussing with a doctor, "
                        "especially together with any symptoms.")

    if values["age"] >= 45:
        personal.append("Heart risk rises with age, so check blood pressure, sugar and "
                        "cholesterol regularly.")

    general = [
        "Aim for about 150 minutes of moderate exercise per week, such as brisk walking.",
        "Eat plenty of vegetables, fruit, pulses and whole grains; limit fried, processed "
        "and sugary food.",
        "Do not smoke, and keep alcohol low.",
        "Sleep 7 to 8 hours and find ways to manage stress.",
    ]
    emergency = ("Get emergency medical help immediately for chest pain, pain spreading to the "
                 "arm or jaw, severe breathlessness or fainting.")
    return {"personal": personal, "general": general, "emergency": emergency}
