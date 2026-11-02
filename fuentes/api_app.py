import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="API de Credit Scoring - XGBoost GPU",
    description="Servicio de inferencia en tiempo real para evaluacion de riesgo crediticio",
    version="1.0.0"
)

# Cargar el modelo de forma absoluta y segura dentro del contenedor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"[INFO] Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    model_pipeline = None
    print(f"[ERROR CRITICO] Fallo al cargar el modelo en {MODEL_PATH}: {str(e)}")

class SolicitudCredito(BaseModel):
    revolving_utilization: float
    age: int
    debt_ratio: float
    monthly_income: float
    open_credit_lines: int
    times_90_days_late: int
    number_dependents: int

@app.get("/")
def read_root():
    return {"status": "Online", "model": "XGBoost Credit Scoring", "threshold": 0.44}

@app.post("/predict")
def predict_risk(solicitud: SolicitudCredito):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="El modelo no se encuentra cargado en el servidor.")
    
    # Crear un DataFrame con TODAS las columnas que el pipeline de scikit-learn espera
    input_data = pd.DataFrame([{
        'Unnamed: 0': 0, # Columna índice que a veces se arrastra del CSV
        'RevolvingUtilizationOfUnsecuredLines': solicitud.revolving_utilization,
        'age': solicitud.age,
        'NumberOfTime30-59DaysPastDueNotWorse': 0, # Valor por defecto o ajustable
        'DebtRatio': solicitud.debt_ratio,
        'MonthlyIncome': solicitud.monthly_income,
        'NumberOfOpenCreditLinesAndLoans': solicitud.open_credit_lines,
        'NumberOfTimes90DaysLate': solicitud.times_90_days_late,
        'NumberRealEstateLoansOrLines': 0,        # Valor por defecto o ajustable
        'NumberOfTime60-89DaysPastDueNotWorse': 0, # Valor por defecto o ajustable
        'NumberOfDependents': solicitud.number_dependents
    }])
    
    try:
        probabilidad_fraude = float(model_pipeline.predict_proba(input_data)[0][1])
        umbral_optimo = 0.44
        prediccion = 1 if probabilidad_fraude >= umbral_optimo else 0
        resultado_str = "Fraude / Alto Riesgo" if prediccion == 1 else "Legitima / Solvente"
        
        return {
            "prediccion_clase": prediccion,
            "resultado": resultado_str,
            "probabilidad_default": round(probabilidad_fraude, 4),
            "probabilidad_legitima": round(1.0 - probabilidad_fraude, 4),
            "umbral_utilizado": umbral_optimo
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la inferencia: {str(e)}")