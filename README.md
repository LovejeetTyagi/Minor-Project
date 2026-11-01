# Heart Disease Prediction System

B.Tech CSE semester minor project. A Flask web app that predicts the likelihood of heart disease
from 13 clinical features using a scikit-learn model trained on the UCI Heart Disease dataset.

> Educational use only. This is not a medical diagnostic tool.

## Project structure
```
heart-disease-prediction/
├── app.py                  Flask app (pages + JSON API)
├── inspect_data.py         Quick dataset inspection
├── requirements.txt        Packages to run the app
├── requirements-dev.txt    Extra packages to run the notebook
├── Procfile                Deployment command (gunicorn)
├── dataset/heart.csv       Dataset (add this yourself)
├── models/                 heart_disease_model.pkl, scaler.pkl (created by the notebook)
├── notebooks/model_training.ipynb
├── templates/              base, index, predict, result, about, error
└── static/                 css/style.css, js/main.js, images/
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements-dev.txt
```

1. Put the Kaggle `heart.csv` in `dataset/` (13 feature columns + `target`).
2. `python inspect_data.py` to check the data.
3. Open `notebooks/model_training.ipynb` and **Run All**. This creates the two files in `models/`.
4. `python app.py` and open http://127.0.0.1:5000

## API
`POST /api/predict` with JSON containing all 13 features:
```json
{"age":54,"sex":1,"cp":0,"trestbps":130,"chol":240,"fbs":0,"restecg":1,
 "thalach":150,"exang":1,"oldpeak":1.5,"slope":1,"ca":1,"thal":3}
```
Other endpoints: `GET /health`, `GET /api/info`.

## Deployment (Render / Railway / Heroku)
Commit `models/*.pkl`, then deploy with the `Procfile` (`web: gunicorn app:app`) and `requirements.txt`.
Use the same scikit-learn version for training and serving, otherwise loading the model can fail.

## Features
Age, sex, chest pain type (cp), resting BP (trestbps), cholesterol (chol), fasting blood sugar (fbs),
resting ECG (restecg), max heart rate (thalach), exercise angina (exang), ST depression (oldpeak),
ST slope (slope), vessels coloured (ca), thalassemia (thal).
