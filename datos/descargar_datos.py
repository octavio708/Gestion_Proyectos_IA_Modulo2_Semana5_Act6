import os
import pandas as pd

def verificar_o_preparar_datos():
    directorio_destino = "datos/datos_ini"
    archivo_csv = os.path.join(directorio_destino, "credit_scoring.csv")
    
    # 1. Asegurar que la carpeta de destino exista
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        print(f"[INFO] Carpeta creada exitosamente: {directorio_destino}")
    
    # 2. Verificar si el archivo CSV del dataset ya se encuentra en la ruta
    if os.path.exists(archivo_csv):
        print(f"[✔] El dataset ya se encuentra disponible en: {archivo_csv}")
        df = pd.read_csv(archivo_csv)
        print(f"    - Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas.")
        
        # 3. Mapeo y validación automática de la columna objetivo
        if 'SeriousDlqin2yrs' in df.columns:
            print("    - Validación superada: Columna objetivo 'SeriousDlqin2yrs' encontrada (Give Me Some Credit).")
        elif 'target_default' in df.columns:
            print("    - Validación superada: Columna objetivo 'target_default' ya se encuentra configurada.")
        else:
            print("    [ADVERTENCIA] El archivo no contiene una columna objetivo válida.")
    else:
        print(f"[!] No se encontró el archivo 'credit_scoring.csv' en '{directorio_destino}'.")

if __name__ == "__main__":
    verificar_o_preparar_datos()