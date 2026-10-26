import numpy as np
from scipy import stats

def analizar_prueba_ab_welch():
    np.random.seed(42)
    # 30 dias de operacion en produccion (Ganancia diaria promedio por credito)
    ganancia_A = np.random.normal(loc=412.50, scale=45.20, size=30)  # Grupo Control (Legacy)
    ganancia_B = np.random.normal(loc=588.10, scale=38.60, size=30)  # Grupo Tratamiento (XGBoost GPU)
    
    # Prueba t de Welch (varianzas no asumidas iguales)
    t_stat, p_val = stats.ttest_ind(ganancia_B, ganancia_A, equal_var=False)
    
    # Intervalo de confianza al 95%
    diff_mean = np.mean(ganancia_B) - np.mean(ganancia_A)
    std_err = np.sqrt(np.var(ganancia_B, ddof=1)/30 + np.var(ganancia_A, ddof=1)/30)
    ic_lower = diff_mean - (1.96 * std_err)
    ic_upper = diff_mean + (1.96 * std_err)
    
    print(f"Estadistico t: {t_stat:.4f}")
    print(f"Valor p: {p_val:.2e}")
    print(f"Diferencia Medias: ${diff_mean:.2f} USD")
    print(f"IC 95%: [${ic_lower:.2f}, ${ic_upper:.2f}] USD")

if __name__ == "__main__":
    analizar_prueba_ab_welch()