import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def cargar_y_preprocesar_datos(filepath):
    df = pd.read_csv(filepath)
    
    # Mapeo automático de la columna objetivo oficial de Give Me Some Credit
    if 'SeriousDlqin2yrs' in df.columns:
        df = df.rename(columns={'SeriousDlqin2yrs': 'target_default'})
        print("[INFO] Dataset detectado: Give Me Some Credit ('SeriousDlqin2yrs' renombrado a 'target_default')")
    else:
        raise ValueError("[ERROR] No se encontró la columna objetivo 'SeriousDlqin2yrs' en el CSV.")

    # Limpieza de valores nulos (comunes en MonthlyIncome y NumberOfDependents en este dataset)
    df = df.dropna()

    # Separar características (X) y variable objetivo (y)
    # (Se excluye la primera columna de índice si el CSV la trae guardada)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    X = df.drop(columns=['target_default'])
    y = df['target_default']
    
    # División estratificada inicial (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Identificación de columnas numéricas
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Preprocesador con RobustScaler para aislar valores atípicos sin data leakage
    preprocessor = ColumnTransformer(
        transformers=[('num', RobustScaler(), num_cols)]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor