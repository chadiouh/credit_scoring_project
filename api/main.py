from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

# === Initialisation FastAPI ===
app = FastAPI(title="Credit Scoring API", description="API de prédiction LightGBM optimisé", version="1.0")

# === Chargement des artefacts ===
model = joblib.load("model.pkl")
imputer = joblib.load("preprocessor.pkl")

with open("baseline_row.json", "r") as f:
    baseline_row = json.load(f)

with open("top_features.json", "r") as f:
    top_features = json.load(f)

# === Définition du seuil optimal ===
BEST_THRESHOLD = 0.42  # Remplacer par le seuil trouvé (à ajuster)

# === Fonction métier de coût ===
def cost_score(y_true, y_pred, cost_fn=10, cost_fp=1):
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    tp = cm.loc[1, 1] if (1, 1) in cm.values else 0
    fp = cm.loc[0, 1] if (0, 1) in cm.values else 0
    fn = cm.loc[1, 0] if (1, 0) in cm.values else 0
    return fn * cost_fn + fp * cost_fp

# === Classe d'entrée via Pydantic ===
class InputData(BaseModel):
    data: dict

# === Route d’accueil ===
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Credit Scoring"}

# === Route de prédiction ===
@app.post("/predict")
def predict(input: InputData):
    # Compléter les features manquantes avec baseline
    input_data = input.data
    complete_row = baseline_row.copy()
    complete_row.update(input_data)

    # Conversion en DataFrame
    X = pd.DataFrame([complete_row])

    # Prétraitement
    X_imputed = imputer.transform(X)

    # Prédiction proba
    y_proba = model.predict_proba(X_imputed)[:, 1][0]
    y_pred = int(y_proba >= BEST_THRESHOLD)

    return {
        "proba": round(y_proba, 4),
        "prediction": y_pred,
        "threshold": BEST_THRESHOLD
    }
