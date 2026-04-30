# =====================================================
# COMPARACIÓN PROFESIONAL DE MODELOS CON VALIDACIÓN CRUZADA
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 1. CARGAR DATASET
# ==============================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path   = os.path.join(current_dir, "diabetes.csv")
data        = pd.read_csv(file_path)

# ==============================
# 2. PREPROCESAMIENTO: CEROS INVÁLIDOS
# ==============================
# En el dataset Pima Indians varios campos tienen ceros que son
# biológicamente imposibles (nadie tiene glucosa = 0 o IMC = 0).
# Representan datos faltantes que se registraron como cero.
# Los reemplazamos por NaN y luego imputamos con la mediana,
# calculada SOLO sobre los valores no-cero para no sesgar la estimación.
#
# Ceros detectados por columna:
#   Glucose       →   5 ceros  (  0.7%) → imputar con mediana
#   BloodPressure →  35 ceros  (  4.6%) → imputar con mediana
#   SkinThickness → 227 ceros  ( 29.6%) → imputar con mediana por clase
#   Insulin       → 374 ceros  ( 48.7%) → imputar con mediana por clase
#   BMI           →  11 ceros  (  1.4%) → imputar con mediana
#
# SkinThickness e Insulin tienen >25% de ceros: la imputación simple
# con mediana global introduciría demasiado sesgo. Se imputa por clase
# (diabético vs no diabético) para preservar mejor la distribución real.

# -- Columnas con imputación simple (pocos ceros) --
cols_simple = ['Glucose', 'BloodPressure', 'BMI']

# -- Columnas con imputación por clase (muchos ceros) --
cols_por_clase = ['SkinThickness', 'Insulin']

print("\n========== PREPROCESAMIENTO ==========\n")

# Marcar ceros como NaN
for col in cols_simple + cols_por_clase:
    n_ceros = (data[col] == 0).sum()
    data[col] = data[col].replace(0, np.nan)
    print(f"  {col:<16}: {n_ceros} ceros reemplazados por NaN")

# Imputar columnas simples con mediana global (excluyendo NaN)
for col in cols_simple:
    mediana = data[col].median()
    data[col] = data[col].fillna(mediana)
    print(f"  {col:<16}: imputado con mediana global = {mediana:.1f}")

# Imputar columnas problemáticas con mediana por clase
for col in cols_por_clase:
    for clase in [0, 1]:
        mediana_clase = data.loc[data['Outcome'] == clase, col].median()
        mask = (data['Outcome'] == clase) & (data[col].isna())
        data.loc[mask, col] = mediana_clase
        label = "no diabético" if clase == 0 else "diabético"
        print(f"  {col:<16}: clase {clase} ({label}) → mediana = {mediana_clase:.1f}")

print("\nPreprocesamiento completado. Ceros restantes en columnas tratadas:")
for col in cols_simple + cols_por_clase:
    print(f"  {col:<16}: {(data[col] == 0).sum()} ceros")

# ==============================
# 3. SEPARAR FEATURES Y TARGET
# ==============================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==============================
# 4. DIVISIÓN TRAIN / TEST FINAL
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 5. DEFINIR MODELOS (PIPELINE)
# ==============================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),

    "MLP Neural Network": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(50, 30),
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42
        ))
    ])
}

# ==============================
# 6. VALIDACIÓN CRUZADA (5-FOLD)
# ==============================

scoring = ['accuracy', 'precision', 'recall', 'f1']
results = {}

print("\n========== VALIDACIÓN CRUZADA (5-FOLD) ==========\n")

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )

    print(f"Modelo: {name}")
    print(f"  Accuracy Promedio:  {scores['test_accuracy'].mean():.4f}")
    print(f"  Precision Promedio: {scores['test_precision'].mean():.4f}")
    print(f"  Recall Promedio:    {scores['test_recall'].mean():.4f}")
    print(f"  F1-score Promedio:  {scores['test_f1'].mean():.4f}")
    print("-------------------------------------------------\n")

    results[name] = scores['test_recall'].mean()  # Métrica principal: Recall

# ==============================
# 7. SELECCIÓN DEL MEJOR MODELO
# ==============================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Modelo seleccionado (según Recall promedio): {best_model_name}")

# ==============================
# 8. ENTRENAR Y CALIBRAR MEJOR MODELO
# ==============================

best_model.fit(X_train, y_train)

# Calibrar probabilidades con isotonic regression
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

y_pred = calibrated_model.predict(X_test)

print("\n========== EVALUACIÓN FINAL EN TEST ==========\n")
print(classification_report(y_test, y_pred))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 9. GUARDAR MODELO GANADOR (CALIBRADO)
# ==============================

model_path = os.path.join(current_dir, "best_model.pkl")
joblib.dump(calibrated_model, model_path)
print("\nModelo calibrado guardado en:", model_path)
print("Modelo guardado como 'best_model.pkl'")
