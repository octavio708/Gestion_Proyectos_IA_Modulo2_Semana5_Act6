import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def entrenar_evaluar_modelos_gpu(X_train, X_test, y_train, y_test, preprocessor):
    # Forzar el almacenamiento local en la raíz del proyecto
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Credit_Scoring_Evaluation_GPU")
    
    # 1. BASELINE: Regresion Logistica en CPU
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        pipeline_base = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        pipeline_base.fit(X_train, y_train)
        y_pred_base = pipeline_base.predict(X_test)
        y_prob_base = pipeline_base.predict_proba(X_test)[:, 1]
        
        mlflow.log_param("clasificador", "LogisticRegression")
        mlflow.log_param("clasificador__class_weight", "balanced")
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob_base))
        generar_y_guardar_cm(y_test, y_pred_base, "cm_baseline.png", "Logistic Regression Baseline")
        mlflow.log_artifact("cm_baseline.png")

    # 2. MODELO CANDIDATO: XGBoost Optimizado en GPU (NVIDIA RTX 4070 Ti)
    with mlflow.start_run(run_name="XGBoost_GPU_Optimized"):
        pipeline_xgb = Pipeline([
            ('prep', preprocessor),
            ('clf', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=8.33,  
                tree_method='hist',    
                device='cuda',          
                random_state=42
            ))
        ])
        print("[INFO] Ejecutando entrenamiento acelerado por GPU CUDA...")
        pipeline_xgb.fit(X_train, y_train)
        print("[INFO] Entrenamiento finalizado exitosamente en GPU.")
        
        y_prob_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
        
        # Optimización y evaluación de umbrales por costos
        p_optimo, costo_minimo = optimizar_umbral_costo(y_test, y_prob_xgb)
        y_pred_xgb_opt = (y_prob_xgb >= p_optimo).astype(int)
        
        # Generar curva ROC comparativa
        graficar_curva_roc(y_test, y_prob_base, y_prob_xgb)
        
        # Métricas y Parámetros en MLflow
        mlflow.log_param("hardware_device", "cuda:0 (NVIDIA RTX 4070 Ti)")
        mlflow.log_param("clasificador__learning_rate", 0.1)
        mlflow.log_param("clasificador__max_depth", 6)
        mlflow.log_param("clasificador__scale_pos_weight", 8.33)
        mlflow.log_param("umbral_optimo_p", p_optimo)
        
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob_xgb))
        mlflow.log_metric("costo_operativo_minimo_usd", costo_minimo)
        
        # Registro de artefactos gráficos
        generar_y_guardar_cm(y_test, y_pred_xgb_opt, "cm_xgboost_gpu.png", f"XGBoost GPU (p*={p_optimo:.2f})")
        mlflow.log_artifact("cm_xgboost_gpu.png")
        mlflow.log_artifact("threshold_tuning_plot.png")
        mlflow.log_artifact("curva_roc_comparativa.png")

def generar_y_guardar_cm(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {title}')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Valor Real (Ground Truth)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def optimizar_umbral_costo(y_true, y_probs, costo_fn=10000, costo_fp=1200):
    umbrales = np.linspace(0.01, 0.99, 99)
    costos = []
    
    for p in umbrales:
        y_pred = (y_probs >= p).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costo = (fn * costo_fn) + (fp * costo_fp)
        costos.append(costo)
        
    idx_opt = np.argmin(costos)
    p_optimo = umbrales[idx_opt]
    
    plt.figure(figsize=(8, 4))
    plt.plot(umbrales, np.array(costos)/1e6, color='crimson', lw=2)
    plt.axvline(x=p_optimo, color='navy', linestyle='--', label=f'p* Óptimo = {p_optimo:.2f}')
    plt.title('Minimización de Pérdida Económica por Umbral')
    plt.xlabel('Umbral de Probabilidad (p)')
    plt.ylabel('Pérdida Total (Millones USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("threshold_tuning_plot.png", dpi=300)
    plt.close()
    
    return p_optimo, costos[idx_opt]

def graficar_curva_roc(y_test, y_prob_base, y_prob_xgb):
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
    fpr_x, tpr_x, _ = roc_curve(y_test, y_prob_xgb)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_b, tpr_b, label=f'Baseline LogReg (AUC = {auc(fpr_b, tpr_b):.4f})', color='gray')
    plt.plot(fpr_x, tpr_x, label=f'XGBoost GPU (AUC = {auc(fpr_x, tpr_x):.4f})', color='dodgerblue', lw=2)
    plt.plot([0, 1], [0, 1], 'r--', label='Clasificador Aleatorio (AUC = 0.50)')
    plt.title('Comparativa de Curvas ROC (Evaluation)')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR / Recall)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("curva_roc_comparativa.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    print("[INFO] Cargando dataset para entrenamiento...")
    df = pd.read_csv("datos/datos_ini/credit_scoring.csv")
    
    # Mapeo de columna objetivo si es necesario
    if 'SeriousDlqin2yrs' in df.columns:
        df = df.rename(columns={'SeriousDlqin2yrs': 'target_default'})

    X = df.drop(columns=['target_default'])
    y = df['target_default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocesador básico (imputación y escalado)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ]
    )

    print("[INFO] Iniciando pipeline de entrenamiento y evaluación en GPU...")
    entrenar_evaluar_modelos_gpu(X_train, X_test, y_train, y_test, preprocessor)
    print("[✔] Proceso de entrenamiento y registro en MLflow completado con éxito.")