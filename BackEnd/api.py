from fastapi import FastAPI
import joblib
import numpy as np
import os

# Crear app
app = FastAPI()

# ==============================
# CARGAR MODELO
# ==============================

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_model.pkl")

model = joblib.load(model_path)

# ==============================
# ENDPOINT PRINCIPAL
# ==============================

@app.get("/")
def home():
    return {"mensaje": "API de predicción de diabetes funcionando"}

# ==============================
# ENDPOINT DE PREDICCIÓN
# ==============================


@app.post("/predict")
def predict(data: dict):

    try:
        # Extraer datos del frontend
        pregnancies = data["pregnancies"]
        glucose = data["glucose"]
        blood_pressure = data["blood_pressure"]
        skin_thickness = data["skin_thickness"]
        insulin = data["insulin"]
        bmi = data["bmi"]
        dpf = data["diabetes_pedigree_function"]
        age = data["age"]

        # Convertir a formato modelo
        input_data = np.array([[ 
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        # Predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        probability = float(probability)
        probability = round(probability, 4)

        if probability < 0.3:
            riesgo = "Bajo riesgo de diabetes"
        elif probability < 0.7:
            riesgo = "Riesgo moderado de diabetes"
        else:
            riesgo = "Alto riesgo de diabetes"
            

        return {
    "prediction": int(prediction),
    "probability": float(probability),
    "risk": riesgo

    
}

    except Exception as e:
        return {"error": str(e)}