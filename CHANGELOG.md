# Registro de Cambios (Changelog)

Todas las modificaciones notables de este proyecto seran documentadas en este archivo.

## [1.0.0] - 2026-10-26

### Añadido
- Implementacion del script de entrenamiento (`train.py`) con soporte para GPU CUDA (`device='cuda'`, `tree_method='hist'`).
- Integracion de MLflow v3.17.2 para el registro de parametros, metricas (`roc_auc`, costos operativos) y artefactos graficos.
- Modulo de optimizacion economica de umbrales de decision ($p^* = 0.44$) para minimizar perdidas por Falsos Negativos y Falsos Positivos.
- Generacion automatizada y registro en MLflow de matrices de confusion, curva ROC comparativa (Baseline LogReg AUC 0.8020 vs XGBoost GPU AUC 0.8674) y grafica de sensibilidad de costos.

### Correcciones
- Estandarizacion de rutas locales y habilitacion de la variable de entorno `MLFLOW_ALLOW_FILE_STORE=true` para garantizar la persistencia y lectura correcta de los registros en Windows.
- Resolucion de conflictos de sockets de red en Uvicorn al iniciar la interfaz web de MLflow mediante el enlace explicito a `127.0.0.1`.