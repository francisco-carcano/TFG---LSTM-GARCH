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


def predict_next_price_lstm(model, df, scaler_X, scaler_y, lookback=12, feature_cols=['Open', 'High', 'Low', 'Close'], prnt=True):
    """
    Realiza una predicción del siguiente precio de cierre usando un modelo LSTM ya cargado.
    """
    df = df.sort_values('Datetime').reset_index(drop=True)
    last_window = df[feature_cols].iloc[-lookback:].copy()

    if (last_window <= 0).any().any():
        raise ValueError("Existen valores <= 0 en las features. No se puede aplicar log().")

    last_window_log = np.log(last_window.values)
    last_window_scaled = scaler_X.transform(last_window_log).reshape(1, lookback, len(feature_cols))

    pred_scaled = model.predict(last_window_scaled, verbose=0)
    pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    predicted_price = np.exp(pred_log)

    if prnt:
        print(f"📈 Predicción del cierre de la próxima vela: {predicted_price:.5f}")

    return predicted_price


def get_realized_volatility(model, horizon=1, scale_factor=1e4, prnt=True):
    """
    Calcula la desviación estándar real (desescalada) del log-retorno
    predicha por un modelo de volatilidad (GARCH, FIGARCH, etc.).
    """
    predicted_var_scaled = predict_volatility(model, horizon=horizon)[0]
    var_real = predicted_var_scaled / (scale_factor ** 2)
    std_real = np.sqrt(var_real)

    if prnt:
        print(f"📉 Desviación estándar real del log-retorno: {std_real:.8f}")
    return std_real

#actualizaBD()
# Cargar los datos
ruta_datos = './Data/eurusd_5m_final.xlsx'
df = read_df(ruta_datos, mostrar_info=False)
df = target(df)

# --------- ESCALADO Y CARGA DE SCALERS ---------
# Escalar datos para recuperar los scalers originales
scaler_X = joblib.load('./models/LSTM/scaler_X.pkl')
scaler_y = joblib.load('./models/LSTM/scaler_y.pkl')

# --------- CARGA Y ACTUALIZACIÓN DEL MODELO LSTM ---------
#lstm_model_path = r'.\models\LSTM\ConCapas\lstm_u256_lr0.00020_sgd_bidi copy.keras'
if df['Target'].isna().sum() > 0:
    df = df.dropna(subset=['Target']).reset_index(drop=True)

#lstm_model = load_model(lstm_model_path)

# days_to_update = 14
# lstm_model = update_model_with_last_days(
#     model_path=lstm_model_path,
#     df=df,
#     scaler_X=scaler_X,
#     scaler_y=scaler_y,
#     days=days_to_update,
#     save_path=lstm_model_path  # Sobrescribe el modelo original
# )
# print("Modelo LSTM actualizado con datos recientes")

# --------- PREDICCIÓN DEL SIGUIENTE CIERRE ---------
#predicted_price = predict_next_price_lstm(lstm_model, df, scaler_X, scaler_y, prnt=False)

# --------- CONFIGURACIÓN MODELO GARCH ---------
# Calcular log-retornos
log_returns = log_return(df)

# Ajustar un modelo FIGARCH(1,1)
figarh_model = fit_figarch_model(log_returns,1,1,0.5)
std_real = get_realized_volatility(figarh_model, prnt=False)

# --------- BackTesting ---------
def walk_forward_backtest_with_retraining(df,
                                          model_path,
                                          garch_model,
                                          scaler_X,
                                          scaler_y,
                                          lookback=12,
                                          retrain_interval_horas=12,
                                          ventana_entrenamiento_dias=3,
                                          backtest_dias=14,
                                          k=2,
                                          moving_tp_sl=False):
    """
    Backtesting walk-forward: reentrena cada N horas usando últimos D días y predice la siguiente vela.
    """
    df = df.sort_values('Datetime').reset_index(drop=True)
    inicio_backtest = df['Datetime'].max() - pd.Timedelta(days=backtest_dias)
    df_backtest = df[df['Datetime'] >= inicio_backtest].reset_index(drop=True)

    modelo_actual = load_model(model_path)
    aciertos, fallos, operaciones, ganancia_total_pips = 0, 0, 0, 0

    hora_ultima_actualizacion = None
    coste_operacion_pips = 0.3  # Comisión fija del broker (spread + fees)

    for i in tqdm(range(lookback, len(df_backtest) - 1), desc="Walk-forward backtesting"):
        ahora = df_backtest['Datetime'].iloc[i]

        if (hora_ultima_actualizacion is None) or ((ahora - hora_ultima_actualizacion).total_seconds() >= retrain_interval_horas * 3600):
            hora_ultima_actualizacion = ahora
            cutoff = ahora - pd.Timedelta(days=ventana_entrenamiento_dias)
            df_entreno = df[df['Datetime'] < ahora]
            df_entreno = df_entreno[df_entreno['Datetime'] >= cutoff]

            try:
                modelo_actual = update_model_with_last_days(
                    model_path=model_path,
                    df=df_entreno,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    days=None,
                    save_path=None,
                    epochs=3
                )
            except Exception as e:
                print(f"❌ Error al reentrenar en {ahora}: {e}")
                continue

        # Predicción
        try:
            predicted_price = predict_next_price_lstm(
                modelo_actual,
                df_backtest.iloc[:i+1],
                scaler_X,
                scaler_y,
                lookback=lookback,
                prnt=False
            )
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            continue

        current_close = df_backtest['Close'].iloc[i]
        direction = np.sign(predicted_price - current_close)
        delta = abs(predicted_price - current_close)
        if direction == 0 or delta < 0.0025:
            continue

        # Volatilidad
        log_returns = log_return(df_backtest.iloc[:i+1])
        garch_model = fit_figarch_model(log_returns, 1, 1, 0.5)
        std_real = get_realized_volatility(garch_model, prnt=False)

        tp = current_close + direction * k * std_real
        sl = current_close - direction * k * std_real

        cerrada = False
        for j in range(i+1, len(df_backtest)):
            high = df_backtest['High'].iloc[j]
            low = df_backtest['Low'].iloc[j]
            precio_actual = df_backtest['Close'].iloc[j]

            if moving_tp_sl:
                movimiento = (precio_actual - current_close) * direction
                if movimiento > 0:
                    tp += direction * movimiento
                    sl += direction * movimiento

            if direction == 1:  # LONG
                if high >= tp:
                    cierre = tp
                    cerrada = True
                    break
                elif low <= sl:
                    cierre = sl
                    cerrada = True
                    break
            else:  # SHORT
                if low <= tp:
                    cierre = tp
                    cerrada = True
                    break
                elif high >= sl:
                    cierre = sl
                    cerrada = True
                    break

        if cerrada:
            pips = (cierre - current_close) * 10000 * direction
            pips -= coste_operacion_pips  # Aplicamos la comisión por operación
            ganancia_total_pips += pips
            operaciones += 1
            if pips > 0:
                aciertos += 1
            else:
                fallos += 1

    print(f"\n📈 Walk-forward backtesting finalizado con {operaciones} operaciones")
    if operaciones > 0:
        print(f"✔️ Aciertos: {aciertos} ({(aciertos / operaciones) * 100:.2f}%)")
    print(f"❌ Fallos: {fallos}")
    print(f"📊 Pips ganados (netos): {ganancia_total_pips:.2f}")
    print(f"💰 Ganancia estimada: {ganancia_total_pips*5:.2f} €")


lstm_model_path = r'.\models\LSTM\ConCapas\lstm_u256_lr0.00020_sgd_bidi copy.keras'
print("Sin moving tp y sl:")

walk_forward_backtest_with_retraining(
    df=df,
    model_path=lstm_model_path,
    garch_model=figarh_model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    lookback=12,
    retrain_interval_horas=12,
    ventana_entrenamiento_dias=3,
    backtest_dias=14,
    k=5,
    moving_tp_sl=False
)

print("Con moving tp y sl:")
lstm_model_path = r'.\models\LSTM\ConCapas\lstm_u256_lr0.00020_sgd_bidi copy 2.keras'

walk_forward_backtest_with_retraining(
    df=df,
    model_path=lstm_model_path,
    garch_model=figarh_model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    lookback=12,
    retrain_interval_horas=12,
    ventana_entrenamiento_dias=3,
    backtest_dias=14,
    k=5,
    moving_tp_sl=True
)
# --------- EVALUACIÓN DE MODELOS GUARDADOS (MSE) ---------
# from lib_ml import evaluate_saved_models, windows

# # Preparamos los datos con las últimas 3.000 velas (ajusta según disponibilidad)
# df_eval = df.copy().sort_values('Datetime').tail(3000)
# X_eval, y_eval = windows(df_eval, lookback=12, delay=1)

# # Escalar con los scalers ya cargados
# X_eval_log = np.log(X_eval).reshape(-1, X_eval.shape[2])
# X_eval_scaled = scaler_X.transform(X_eval_log).reshape(X_eval.shape)

# y_eval_log = np.log(y_eval).reshape(-1, 1)
# y_eval_scaled = scaler_y.transform(y_eval_log).flatten()

# # Directorio donde están guardados los modelos
# models_path = './models/LSTM/'

# # Evaluar los modelos
# resultados_mse = evaluate_saved_models(models_path, X_eval_scaled, y_eval_scaled)

# # Mostrar los resultados ordenados por MSE
# print("\n📊 Resultados de evaluación de modelos por MSE:")
# for mse, nombre in resultados_mse:
#     if isinstance(mse, list):  # Aseguramos que sea lista y extraemos el primer valor
#         mse_val = mse[0]
#     else:
#         mse_val = mse
#     print(f"{nombre}: MSE = {mse_val:.6f}")