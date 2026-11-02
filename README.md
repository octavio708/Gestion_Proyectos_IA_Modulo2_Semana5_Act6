# API de Credit Scoring - XGBoost GPU & FastAPI

Servicio de inferencia en tiempo real para la evaluación de riesgo crediticio, desarrollado como parte del **Módulo 2 (Semana 5) - Gestión de Proyectos de Inteligencia Artificial**. El sistema implementa un modelo de Machine Learning basado en **XGBoost** entrenado con aceleración por GPU (CUDA), integrado en una API de alto rendimiento con **FastAPI** y contenerizado mediante **Docker**.

---

## 🚀 Arquitectura y Estructura del Proyecto

```text
Gestion_Proyectos_IA_Modulo2_Semana5_Act6/
├── data/
│   └── credit_data.csv        # Dataset de entrenamiento
├── fuentes/
│   ├── api_app.py             # Código fuente de la API FastAPI y endpoints
│   ├── train.py               # Script de entrenamiento y tracking con MLflow
│   └── model_pipeline.pkl     # Artefacto serializado del modelo y preprocesamiento
├── Dockerfile                 # Definición de la imagen del contenedor
├── requirements.txt           # Dependencias estandarizadas del proyecto
└── README.md                  # Documentación oficial del proyecto
```

---

## 🛠️ Requisitos Técnicos y Dependencias

El proyecto se encuentra estricta y sincronizado bajo las siguientes versiones clave para evitar errores de deserialización e inferencia:

- **Python:** `3.10-slim`
- **Scikit-Learn:** `1.5.2`
- **XGBoost:** `2.0.3`
- **FastAPI & Uvicorn:** `0.110.0` / `0.28.0`
- **Pandas & NumPy:** `2.2.1` / `1.26.4`

---

## 🐳 Instrucciones de Despliegue con Docker

### 1. Construir la imagen Docker (sin caché)
Abre tu terminal en la raíz del repositorio y ejecuta el siguiente comando:
```bash
docker build --no-cache -t credit-scoring-api:latest .
```

### 2. Ejecutar el contenedor
```bash
docker run -d --name credit-scoring-container -p 8000:8000 credit-scoring-api:latest
```

### 3. Verificar el estado de la API
Puedes acceder a la documentación interactiva (Swagger UI) desde tu navegador en:
👉 `http://localhost:8000/docs`

---

## 🧪 Validación de Inferencia (cURL)

Puedes probar el servicio enviando una solicitud POST al endpoint `/predict`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "revolving_utilization": 0.85,
    "age": 42,
    "debt_ratio": 0.45,
    "monthly_income": 5000.0,
    "open_credit_lines": 8,
    "times_90_days_late": 2,
    "number_dependents": 1
  }'
```

**Respuesta esperada del servidor:**
```json
{
  "prediccion_clase": 1,
  "resultado": "Fraude / Alto Riesgo",
  "probabilidad_default": 0.757,
  "probabilidad_legitima": 0.243,
  "umbral_utilizado": 0.44
}
```
*(Nota: Las decisiones de negocio aplican el umbral óptimo económico de default $\text{umbral} = 0.44$).*