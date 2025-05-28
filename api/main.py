from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os

# === Initialisation FastAPI ===
app = FastAPI(title="Credit Scoring API", description="API de prédiction LightGBM optimisé", version="1.0")

# === Détection du chemin absolu du fichier (fonctionne localement et sur Render) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Chargement des artefacts (modèle, imputer, baseline, features importantes) ===
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))

with open(os.path.join(BASE_DIR, "baseline_row.json"), "r") as f:
    baseline_row = json.load(f)

with open(os.path.join(BASE_DIR, "top_features.json"), "r") as f:
    top_features = json.load(f)

# === Définition du seuil optimal (ajusté selon la validation métier) ===
BEST_THRESHOLD = 0.42  # à ajuster si besoin

# === Classe d'entrée attendue par l'API ===
class InputData(BaseModel):
    data: dict

# === Route d’accueil ===
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Credit Scoring"}

# === Route de prédiction ===
@app.post("/predict")
def predict(input: InputData):
    input_data = input.data

    # Complétion automatique avec la ligne baseline
    complete_row = baseline_row.copy()
    complete_row.update(input_data)

    # Conversion en DataFrame
    X = pd.DataFrame([complete_row])

    # Imputation
    X_imputed = imputer.transform(X)

    # Prédiction
    y_proba = model.predict_proba(X_imputed)[:, 1][0]
    y_pred = int(y_proba >= BEST_THRESHOLD)

    return {
        "proba": round(y_proba, 4),
        "prediction": y_pred,
        "threshold": BEST_THRESHOLD
    }