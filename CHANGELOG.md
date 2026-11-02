# Registro de Cambios (Changelog)

Todas las modificaciones notables de este proyecto son documentadas en este archivo.

## \[1.0.0\] - 2026-11-02

### Añadido

* Implementación del pipeline de entrenamiento en `train.py` utilizando aceleración por GPU CUDA (`tree_method='hist'`).

* Integración del servicio de inferencia en tiempo real mediante `FastAPI` en `api_app.py`.

* Configuración y control de experimentos mediante `MLflow`.

* Contenerización robusta basada en `python:3.10-slim` con `Dockerfile` y `requirements.txt`.

### Cambios y Correcciones

* **Rutas Absolutas:** Se corrigió la carga del modelo en `api_app.py` utilizando `BASE_DIR` y `os.path.join` para prevenir errores de búsqueda de archivos relativos al iniciar Uvicorn dentro del contenedor.

* **Sincronización de Versiones:** Estandarización de `scikit-learn==1.5.2` tanto en entorno local como en Docker para evitar advertencias y errores de descompresión de estimadores (*InconsistentVersionWarning*).

* **Mapeo de Características:** Ajuste en el DataFrame de entrada de la API para incluir todas las columnas originales requeridas por el pipeline de preprocesamiento, resolviendo los errores de atributos faltantes.

* **Selección Numérica Segura:** Actualización en `train.py` utilizando `include=['number']` para evitar incompatibilidades de tipos de datos en Pandas y `ColumnTransformer`.