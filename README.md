# Pipeline de MLOps: Credit Scoring con Aceleracion por GPU y MLflow

## 1. Descripcion del Proyecto
Este repositorio contiene el desarrollo de la Actividad 6 (Gestion de Proyectos de Inteligencia Artificial), implementando un pipeline completo de MLOps para el conjunto de datos de riesgo crediticio (*Give Me Some Credit*). El proyecto integra entrenamiento acelerado por hardware dedicado, optimizacion de umbrales basados en costos de negocio y trazabilidad experimental con MLflow.

## 2. Stack Tecnologico
- Lenguaje: Python 3.13
- Modelado: Scikit-Learn, XGBoost (con soporte CUDA y metodo de histograma en GPU).
- Trazabilidad: MLflow v3.17.2.
- Hardware: NVIDIA GeForce RTX 4070 Ti / Intel Core i7-12700.

## 3. Estructura del Repositorio
- `fuentes/`: Contiene los scripts principales de entrenamiento, procesamiento y evaluacion (`train.py`, etc.).
- `datos/`: Directorio de almacenamiento de los conjuntos de datos iniciales (`credit_scoring.csv`).
- `mlruns/`: Almacenamiento local de experimentos, metricas y artefactos registrados por MLflow.

## 4. Instrucciones de Ejecucion
1. Activar el entorno virtual:
   source venv/Scripts/activate
2. Ejecutar el pipeline de entrenamiento acelerado por GPU:
   python fuentes/train.py
3. Levantar la interfaz web de MLflow para consultar metricas y artefactos:
   export MLFLOW_ALLOW_FILE_STORE=true && mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
4. Acceder al navegador en: http://127.0.0.1:5000