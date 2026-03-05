# =====================================================
# COMPARACIÓN PROFESIONAL DE MODELOS CON VALIDACIÓN CRUZADA
# =====================================================

import pandas as pd
import numpy as np
import joblib

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

data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ==============================
# 2. DIVISIÓN TRAIN / TEST FINAL
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 3. CREAR PIPELINES (IMPORTANTE)
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
            hidden_layer_sizes=(50, 30),   # Arquitectura mejorada
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42
        ))
    ])
}

# ==============================
# 4. VALIDACIÓN CRUZADA
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
    print(f"Accuracy Promedio:  {scores['test_accuracy'].mean():.4f}")
    print(f"Precision Promedio: {scores['test_precision'].mean():.4f}")
    print(f"Recall Promedio:    {scores['test_recall'].mean():.4f}")
    print(f"F1-score Promedio:  {scores['test_f1'].mean():.4f}")
    print("-------------------------------------------------\n")

    results[name] = scores['test_recall'].mean()  # Enfocados en Recall

# ==============================
# 5. SELECCIÓN DEL MEJOR MODELO
# ==============================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 Modelo seleccionado (según Recall promedio): {best_model_name}")

# ==============================
# 6. ENTRENAR MEJOR MODELO CON TODO EL TRAIN
# ==============================

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\n========== EVALUACIÓN FINAL EN TEST ==========\n")
print(classification_report(y_test, y_pred))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7. GUARDAR MODELO GANADOR
# ==============================

joblib.dump(best_model, "best_model.pkl")

print("\nModelo guardado como 'best_model.pkl'")