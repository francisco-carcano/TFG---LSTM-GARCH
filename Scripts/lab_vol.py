
import os

# Configura TensorFlow para que no imprima mensajes de INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL

import tensorflow as tf
from lib_vol import *
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from actualizarBD import actualizaBD
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_excel("Data/eurusd_5m_final.xlsx")
log_returns = log_return(df)

# 2. Verificar estacionariedad
#test_stationarity(log_returns)

# 3. Plots
#all_plots(df, num_velas=10000)  

# 4. CORRELACIÓN ENTRE VOLATILIDAD Y CAMBIO DE PRECIO
#corr_vol_price(df, window=30)  
    #Esta función te dará una visión clara de la relación entre la volatilidad y los movimientos de precio, 
    #lo cual es útil para estrategias de gestión de riesgo y ajuste de tamaño de posición.

# 5. COMPARAR MODELOS#
#plot_model_summaries(df, 1, 1) 
    #Gana Figarch, luego GJR-GARCH y luego GARCH

# 6. COMPARAR MEJOR PARAMETRO FIGARCH
results = compare_figarch_models(df,[0,1],[0,1],[0.3,0.5,0.7]) 
    #Lo mejor es 1,1 y da igual la D
print(results)
# 7. CALCULAR MSE, RMSE I MAPE
# real_volatility = log_returns.rolling(window=30).std().dropna()  # Esto es solo un ejemplo de datos reales
# metrics = calculate_metrics(model_figarch, real_volatility, horizon=1)
# print(f"RMSE: {metrics['RMSE']}")
# print(f"MSE: {metrics['MSE']}")
# print(f"MAPE: {metrics['MAPE']}%")

# 8. METRICAS DE TODOS LOS MODELOS
plot_model_metrics(df, p=1, q=1, d=0.5)  
    #Gana el FIGARCH en las 3 métricas

# 9. EJEMPLO PREDICCIÓN PRÒXIMA VOLATILIDAD
# model_figarch = fit_figarch_model(log_returns,1,1,0.5)
# predicted_volatility = predict_volatility(model_figarch, horizon=1)  
# print(f"Predicción de volatilidad para el siguiente paso: {predicted_volatility}")

