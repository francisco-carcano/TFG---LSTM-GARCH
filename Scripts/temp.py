import os

# Desactiva los prints de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Oculta todos los INFO de TensorFlow (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from lib_ml import *
from lib_vol import *
from actualizarBD import actualizaBD
import joblib
from tqdm import tqdm
from lib_ml import evaluate_saved_models, windows
ruta_datos = './Data/eurusd_5m_final.xlsx'
df = read_df(ruta_datos, mostrar_info=False)
df = target(df)

# --------- ESCALADO Y CARGA DE SCALERS ---------
# Escalar datos para recuperar los scalers originales
scaler_X = joblib.load('./models/LSTM/scaler_X.pkl')
scaler_y = joblib.load('./models/LSTM/scaler_y.pkl')

# --------- CARGA Y ACTUALIZACIÓN DEL MODELO LSTM ---------
lstm_model_path = r'.\models\LSTM\ConCapas\lstm_u256_lr0.00020_sgd_bidi.keras'
if df['Target'].isna().sum() > 0:
    df = df.dropna(subset=['Target']).reset_index(drop=True)

# Preparamos los datos con las últimas 3.000 velas (ajusta según disponibilidad)
df_eval = df.copy().sort_values('Datetime').tail(3000)
X_eval, y_eval = windows(df_eval, lookback=12, delay=1)

# Escalar con los scalers ya cargados
X_eval_log = np.log(X_eval).reshape(-1, X_eval.shape[2])
X_eval_scaled = scaler_X.transform(X_eval_log).reshape(X_eval.shape)

y_eval_log = np.log(y_eval).reshape(-1, 1)
y_eval_scaled = scaler_y.transform(y_eval_log).flatten()

# Directorio donde están guardados los modelos
models_path = './models/LSTM/ConCapas/'

# Evaluar los modelos
resultados = evaluate_saved_models(models_path, X_eval_scaled, y_eval_scaled)

# Mostrar los resultados ordenados por MSE
# Evaluar los modelos
resultados = evaluate_saved_models(models_path, X_eval_scaled, y_eval_scaled)

# Mostrar los resultados ordenados por MSE
print("\n📊 Resultados de evaluación de modelos:")
for mse, mae, rmse, mape, nombre in resultados:
    print(f"{nombre}:")
    print(f"  MSE    = {mse:.6f}")
    print(f"  MAE   = {mae:.6f}")
    print(f"  RMSE  = {rmse:.6f}")
    if mape is not None:
        print(f"  MAPE  = {mape:.6f}")
    else:
        print("  MAPE  = No disponible")
    print("-" * 40)  # Separador visual entre modelos
