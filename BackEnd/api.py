from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_model.pkl")
model = joblib.load(model_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"mensaje": "API de predicción de diabetes funcionando"}

# ==============================
# RANGOS CLÍNICOS DE REFERENCIA
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
# ENDPOINT DE PREDICCIÓN
# ==============================

@app.post("/predict")
def predict(data: dict):
    try:
        pregnancies = data["pregnancies"]
        glucose = data["glucose"]
        blood_pressure = data["blood_pressure"]
        skin_thickness = data["skin_thickness"]
        insulin = data["insulin"]
        bmi = data["bmi"]
        dpf = data["diabetes_pedigree_function"]
        age = data["age"]

        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]])

        prediction = model.predict(input_data)[0]
        probability = round(float(model.predict_proba(input_data)[0][1]), 4)

        if probability < 0.2:
            riesgo = "Bajo riesgo de diabetes"
        elif probability < 0.5:
            riesgo = "Riesgo moderado de diabetes"
        else:
            riesgo = "Alto riesgo de diabetes"

        explicacion = generar_explicacion(
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        )

        return {
            "prediction": int(prediction),
            "probability": probability,
            "risk": riesgo,
            "explicacion": explicacion
        }

    except Exception as e:
        return {"error": str(e)}