from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import json
import logging
from datetime import datetime

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# INICIALIZAR APP
# ==============================
app = FastAPI(
    title="Diabetes Predictor API",
    description="API de predicción de diabetes basada en red neuronal MLP",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# CARGAR MODELO
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path  = os.path.join(current_dir, "best_model.pkl")

try:
    model = joblib.load(model_path)
    logger.info(f"Modelo cargado correctamente desde {model_path}")
except FileNotFoundError:
    model = None
    logger.error(f"No se encontró el modelo en {model_path}. Ejecuta compare_models.py primero.")

# ==============================
# MODELO DE ENTRADA CON VALIDACIÓN PYDANTIC
# Cada campo tiene rango mínimo y máximo basado en el dataset Pima Indians.
# FastAPI valida automáticamente antes de llegar al endpoint:
# si un valor está fuera de rango retorna HTTP 422 con mensaje claro.
# ==============================
class DiabetesInput(BaseModel):
    pregnancies:                int   = Field(..., ge=0,    le=17,   description="Número de embarazos (0–17)")
    glucose:                    float = Field(..., ge=44,   le=199,  description="Glucosa en sangre mg/dL (44–199)")
    blood_pressure:             float = Field(..., ge=24,   le=122,  description="Presión diastólica mmHg (24–122)")
    skin_thickness:             float = Field(..., ge=7,    le=99,   description="Grosor pliegue tríceps mm (7–99)")
    insulin:                    float = Field(..., ge=14,   le=846,  description="Insulina μU/mL (14–846)")
    bmi:                        float = Field(..., ge=18.0, le=67.0, description="Índice de Masa Corporal kg/m² (18–67)")
    diabetes_pedigree_function: float = Field(..., ge=0.07, le=2.42, description="Función de pedigrí de diabetes (0.07–2.42)")
    age:                        int   = Field(..., ge=21,   le=81,   description="Edad en años (21–81)")

# ==============================
# RANGOS CLÍNICOS Y EXPLICACIONES
# ==============================
def generar_explicacion(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    factores = []

    # Glucosa
    if glucose >= 140:
        factores.append({"variable": "Glucosa", "valor": glucose, "estado": "alto",
            "mensaje": f"Tu nivel de glucosa ({glucose} mg/dL) está por encima del umbral de diabetes (≥140 mg/dL). Este es el factor de mayor peso en el modelo."})
    elif glucose >= 100:
        factores.append({"variable": "Glucosa", "valor": glucose, "estado": "moderado",
            "mensaje": f"Tu nivel de glucosa ({glucose} mg/dL) está en rango de prediabetes (100–139 mg/dL)."})
    else:
        factores.append({"variable": "Glucosa", "valor": glucose, "estado": "normal",
            "mensaje": f"Tu nivel de glucosa ({glucose} mg/dL) está dentro del rango normal (<100 mg/dL)."})

    # IMC
    if bmi >= 30:
        factores.append({"variable": "IMC", "valor": bmi, "estado": "alto",
            "mensaje": f"Tu IMC ({bmi}) indica obesidad (≥30), lo cual aumenta significativamente el riesgo de diabetes tipo 2."})
    elif bmi >= 25:
        factores.append({"variable": "IMC", "valor": bmi, "estado": "moderado",
            "mensaje": f"Tu IMC ({bmi}) indica sobrepeso (25–29.9), un factor de riesgo moderado."})
    else:
        factores.append({"variable": "IMC", "valor": bmi, "estado": "normal",
            "mensaje": f"Tu IMC ({bmi}) está en rango saludable (<25)."})

    # Edad
    if age >= 45:
        factores.append({"variable": "Edad", "valor": age, "estado": "alto",
            "mensaje": f"Tu edad ({age} años) es un factor de riesgo importante; la incidencia de diabetes aumenta significativamente a partir de los 45 años."})
    elif age >= 35:
        factores.append({"variable": "Edad", "valor": age, "estado": "moderado",
            "mensaje": f"Tu edad ({age} años) representa un riesgo moderado. Se recomienda monitoreo regular."})
    else:
        factores.append({"variable": "Edad", "valor": age, "estado": "normal",
            "mensaje": f"Tu edad ({age} años) no representa un factor de riesgo elevado por sí sola."})

    # Presión arterial
    if blood_pressure >= 90:
        factores.append({"variable": "Presión Arterial", "valor": blood_pressure, "estado": "alto",
            "mensaje": f"Tu presión diastólica ({blood_pressure} mmHg) está elevada (≥90 mmHg), lo cual se asocia con resistencia a la insulina."})
    elif blood_pressure >= 80:
        factores.append({"variable": "Presión Arterial", "valor": blood_pressure, "estado": "moderado",
            "mensaje": f"Tu presión diastólica ({blood_pressure} mmHg) está en el límite alto (80–89 mmHg)."})
    else:
        factores.append({"variable": "Presión Arterial", "valor": blood_pressure, "estado": "normal",
            "mensaje": f"Tu presión arterial ({blood_pressure} mmHg) está en rango normal."})

    # DPF
    if dpf >= 0.8:
        factores.append({"variable": "Historial Familiar", "valor": dpf, "estado": "alto",
            "mensaje": f"Tu función de pedigrí ({dpf}) es alta, lo que indica una fuerte carga genética familiar de diabetes."})
    elif dpf >= 0.4:
        factores.append({"variable": "Historial Familiar", "valor": dpf, "estado": "moderado",
            "mensaje": f"Tu función de pedigrí ({dpf}) indica un historial familiar moderado."})
    else:
        factores.append({"variable": "Historial Familiar", "valor": dpf, "estado": "normal",
            "mensaje": f"Tu función de pedigrí ({dpf}) sugiere bajo riesgo genético familiar."})

    # Insulina
    if insulin > 200:
        factores.append({"variable": "Insulina", "valor": insulin, "estado": "alto",
            "mensaje": f"Tu nivel de insulina ({insulin} μU/mL) está considerablemente elevado, lo que puede indicar resistencia a la insulina."})
    elif insulin > 100:
        factores.append({"variable": "Insulina", "valor": insulin, "estado": "moderado",
            "mensaje": f"Tu nivel de insulina ({insulin} μU/mL) está algo elevado (normal en ayunas: <100 μU/mL)."})
    else:
        factores.append({"variable": "Insulina", "valor": insulin, "estado": "normal",
            "mensaje": f"Tu nivel de insulina ({insulin} μU/mL) está dentro del rango normal."})

    # Embarazos
    if pregnancies >= 6:
        factores.append({"variable": "Embarazos", "valor": pregnancies, "estado": "alto",
            "mensaje": f"Has tenido {pregnancies} embarazos. Un número elevado se asocia con mayor riesgo de diabetes gestacional y tipo 2."})
    elif pregnancies >= 3:
        factores.append({"variable": "Embarazos", "valor": pregnancies, "estado": "moderado",
            "mensaje": f"Has tenido {pregnancies} embarazos, un factor de riesgo moderado a considerar."})
    else:
        factores.append({"variable": "Embarazos", "valor": pregnancies, "estado": "normal",
            "mensaje": f"El número de embarazos ({pregnancies}) no representa un factor de riesgo elevado."})

    orden = {"alto": 0, "moderado": 1, "normal": 2}
    factores.sort(key=lambda x: orden[x["estado"]])

    return factores

# ==============================
# ENDPOINT RAÍZ
# ==============================
@app.get("/")
def home():
    return {"mensaje": "Diabetes Predictor API v2.0 funcionando"}

# ==============================
# ENDPOINT /health
# Útil para verificar que la API está viva y el modelo está cargado.
# ==============================
@app.get("/health")
def health():
    return {
        "status":        "ok" if model is not None else "error",
        "modelo_cargado": model is not None,
        "timestamp":     datetime.now().isoformat()
    }

# ==============================
# ENDPOINT /model-info
# Expone la versión y métricas del modelo entrenado.
# Lee el archivo model_metrics.json si existe.
# ==============================
@app.get("/model-info")
def model_info():
    metrics_path = os.path.join(current_dir, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {"nota": "model_metrics.json no encontrado. Ejecuta compare_models.py para generarlo."}

    return {
        "modelo":  "MLP Neural Network (128-64-32)",
        "version": "2.0.0",
        "dataset": "Pima Indians Diabetes Dataset",
        "metricas": metrics
    }

# ==============================
# ENDPOINT /predict
# Recibe DiabetesInput (validado por Pydantic) y retorna predicción.
# Si los datos son inválidos, FastAPI retorna HTTP 422 automáticamente.
# Si el modelo no está cargado, retorna HTTP 503.
# ==============================
@app.post("/predict")
def predict(data: DiabetesInput):
    if model is None:
        logger.error("Predicción solicitada pero el modelo no está cargado.")
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta compare_models.py primero.")

    try:
        input_data = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age
        ]])

        prediction  = model.predict(input_data)[0]
        probability = round(float(model.predict_proba(input_data)[0][1]), 4)

        if probability < 0.2:
            riesgo = "Bajo riesgo de diabetes"
        elif probability < 0.5:
            riesgo = "Riesgo moderado de diabetes"
        else:
            riesgo = "Alto riesgo de diabetes"

        explicacion = generar_explicacion(
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree_function, data.age
        )

        logger.info(f"Predicción: prob={probability:.4f} nivel={riesgo}")

        return {
            "prediction":  int(prediction),
            "probability": probability,
            "risk":        riesgo,
            "explicacion": explicacion
        }

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
