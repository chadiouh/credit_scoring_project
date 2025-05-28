import streamlit as st
import requests
import json

# === Titre et description ===
st.title("🧠 Interface Credit Scoring")
st.write("Remplissez les variables principales pour prédire si un client est solvable.")

# === Chargement des top features ===
with open("../models/top_features.json", "r") as f:
    top_features = json.load(f)

# === Création dynamique du formulaire ===
user_input = {}
for feature in top_features:
    user_input[feature] = st.text_input(f"{feature} :")

# === Bouton de prédiction ===
if st.button("Prédire"):
    # Nettoyage et conversion
    try:
        input_data = {k: float(v) for k, v in user_input.items() if v.strip() != ""}
    except ValueError:
        st.error("Toutes les entrées doivent être numériques.")
    else:
        payload = {"data": input_data}
        try:
            response = requests.post("https://credit-scoring-project-1.onrender.com", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Probabilité d'insolvabilité : {result['proba']}")
                st.info(f"Prédiction finale : {'Non solvable' if result['prediction'] == 1 else 'Solvable'}")
                st.caption(f"Seuil utilisé : {result['threshold']}")
            else:
                st.error("Erreur dans l’API : " + response.text)
        except requests.exceptions.ConnectionError:
            st.error("⚠️ L'API n'est pas accessible. Veuillez lancer `main.py` via uvicorn.")
