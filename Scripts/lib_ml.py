import os

# Desactiva los prints de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Oculta todos los INFO de TensorFlow (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt


def read_df(ruta_csv, mostrar_info=True):
    df = pd.read_excel(ruta_csv)
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    if mostrar_info:
        print("\n🔍 Información general del DataFrame:\n")
        print(df.info())
        print("\nEstadísticas descriptivas:\n")
        print(df.describe())
        print("\nValores nulos por columna:\n")
        print(df.isnull().sum())
        print("\nPrimeras filas del DataFrame:\n")
        print(df.head())
    return df

def target(df):
    df2 = df.copy()
    df2['Target'] = df['Close'].shift(-1)    
    df2.loc[df2.index[-1], 'Target'] = float('nan')
    return df2

def target_binary(df):
    df2 = df.copy()
    df2['TargetBin'] = (df2['Close'].shift(-1) > df2['Close']).astype(int)
    df2.loc[df2.index[-1], 'TargetBin'] = np.nan
    return df2

def plot_biweekly_close_mean(df):
    """
    Calcula y grafica la media del precio de cierre cada dos semanas.
    """
    # Asegurarnos de que 'Datetime' esté en datetime y como índice
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Hacemos un resample cada dos semanas ('2W') y calculamos la media de 'Close'
    biweek_means = df['Close'].resample('2W').mean()
    
    # Graficamos
    plt.figure(figsize=(12, 6))
    plt.plot(biweek_means.index, biweek_means.values, marker='o', linestyle='-')
    plt.title("Precio de Cierre Medio Cada 2 Semanas")
    plt.xlabel("Fecha")
    plt.ylabel("Cierre Medio EUR/USD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def windows(df, lookback=12, delay=1,
            feature_cols=['Open','High','Low','Close'],
            target_col='Target'):
    features = df[feature_cols].to_numpy()     
    target   = df[target_col].to_numpy()       
    n_rows, n_features = features.shape
    n_samples = n_rows - lookback - delay + 1
    X = np.empty((n_samples, lookback, n_features), dtype=features.dtype)
    y = np.empty((n_samples,), dtype=target.dtype)
    for i in range(n_samples):
        X[i] = features[i : i + lookback]
        y[i] = target[i + lookback + delay - 1]
    return X, y

def escalar_datos(X, y, feature_range=(-1, 1)):
    X = np.log(X)
    y = np.log(y)
    n_samples, timesteps, n_features = X.shape
    X_2d = X.reshape(-1, n_features)
    scaler_X = MinMaxScaler(feature_range=feature_range)
    X_scaled_2d = scaler_X.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, timesteps, n_features)
    y_reshaped = y.reshape(-1, 1)
    scaler_y = MinMaxScaler(feature_range=feature_range)
    y_scaled = scaler_y.fit_transform(y_reshaped).flatten()
    return X_scaled, y_scaled, scaler_X, scaler_y

def save_scalers(scaler_X, scaler_y, save_dir='saved_scalers'):
    """
    Guarda los scalers en archivos para usarlos más tarde.
    """
    os.makedirs(save_dir, exist_ok=True)    
    joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
    print("Scalers guardados en el directorio:", save_dir)

def split_data(X, y, train_frac=0.8, val_frac=0.2):
    n_samples = X.shape[0]

    train_end = int(n_samples * train_frac)
    X_train_full = X[:train_end]
    y_train_full = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]

    n_train = X_train_full.shape[0]
    split_idx = int(n_train * (1 - val_frac))
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_save_models(
    X_train, y_train, X_val, y_val,
    lookback, n_features,
    units_list,
    lr_list,
    optimizer_list=['adam', 'sgd'],
    epochs=50,
    batch_size=32,
    save_dir='saved_models',
    use_stacked=False,
    use_dense=False,
    use_bidirectional=False
):
    """
    Entrena LSTM con distintas arquitecturas según flags:
      - use_stacked: 2 capas LSTM en serie
      - use_dense: capa Dense intermedia antes de la salida
      - use_bidirectional: LSTM(s) envuelto(s) en Bidirectional
    Guarda modelo y curvas de métricas (train/val) en PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    for units in units_list:
        for lr in lr_list:
            for opt_name in optimizer_list:

                # 1) Construcción del modelo
                model = Sequential()
                model.add(Input(shape=(lookback, n_features)))

                # Primera LSTM (posible return_sequences si apilamos)
                return_seq = use_stacked
                lstm_layer = LSTM(units,
                                  dropout=0.2,
                                  recurrent_dropout=0.2,
                                  return_sequences=return_seq)
                if use_bidirectional:
                    model.add(Bidirectional(lstm_layer))
                else:
                    model.add(lstm_layer)

                # Si apilamos, añadimos segunda LSTM (no necesita return_sequences)
                if use_stacked:
                    lstm_layer2 = LSTM(units,
                                       dropout=0.2,
                                       recurrent_dropout=0.2)
                    if use_bidirectional:
                        model.add(Bidirectional(lstm_layer2))
                    else:
                        model.add(lstm_layer2)

                # Capa dense intermedia opcional
                if use_dense:
                    model.add(Dense(units // 2, activation='relu'))

                # Capa de salida
                model.add(Dense(1, name='output'))

                # 2) Compilación con métricas de interés
                optimizer = Adam(learning_rate=lr) if opt_name=='adam' else SGD(learning_rate=lr)
                model.compile(
                    loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=[
                        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                        tf.keras.metrics.MeanAbsoluteError(name='mae'),
                        tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
                    ]
                )

                # 3) Entrenamiento
                print(f"\nEntrenando: units={units}, lr={lr}, opt={opt_name}"
                      f", stacked={use_stacked}, dense={use_dense}, bidi={use_bidirectional}")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )

                # 4) Guardar modelo
                suffix = f"u{units}_lr{lr:.5f}_{opt_name}" \
                         f"{'_stacked' if use_stacked else ''}" \
                         f"{'_dense' if use_dense else ''}" \
                         f"{'_bidi' if use_bidirectional else ''}"
                model_path = os.path.join(save_dir, f"lstm_{suffix}.keras")
                model.save(model_path)
                print(f"  ↳ Modelo guardado en {model_path}")

                # 5) Graficar métricas en dos imágenes (train y val)
                met_keys = ['loss', 'rmse', 'mae', 'mape']
                titles   = ['MSE (Loss)', 'RMSE', 'MAE', 'MAPE (%)']

                # * Train
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                for ax, key, title in zip(axes.flatten(), met_keys, titles):
                    ax.plot(history.history[key], color='blue', label='train')
                    ax.set_title(title)
                    ax.set_xlabel('Épocas')
                    ax.set_ylabel(title)
                    ax.legend()
                    ax.grid(True)
                plt.suptitle(f"Train metrics: {suffix}", fontsize=16)
                plt.tight_layout(rect=[0,0,1,0.96])
                train_plot = os.path.join(save_dir, f"train_metrics_{suffix}.png")
                fig.savefig(train_plot)
                plt.close(fig)
                print(f"  ↳ Curvas TRAIN guardadas en {train_plot}")

                # * Validation
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                for ax, key, title in zip(axes.flatten(), met_keys, titles):
                    ax.plot(history.history[f"val_{key}"], color='orange', label='val')
                    ax.set_title(title)
                    ax.set_xlabel('Épocas')
                    ax.set_ylabel(title)
                    ax.legend()
                    ax.grid(True)
                plt.suptitle(f"Val metrics: {suffix}", fontsize=16)
                plt.tight_layout(rect=[0,0,1,0.96])
                val_plot = os.path.join(save_dir, f"val_metrics_{suffix}.png")
                fig.savefig(val_plot)
                plt.close(fig)
                print(f"  ↳ Curvas VALIDACIÓN guardadas en {val_plot}")


def evaluate_saved_models(models_dir, X_eval, y_eval):
    results = []
    
    # Reemplazar el NaN al final de y_eval (si es que existe) con el último valor no nulo
    if np.isnan(y_eval[-1]):
        y_eval[-1] = y_eval[-2]  # Asigna el valor de la penúltima vela
    
    # Recorre todas las carpetas y subcarpetas dentro de models_dir
    for root, dirs, files in os.walk(models_dir):
        for fname in files:
            if not fname.endswith('.keras'):
                continue
            
            # Construir la ruta completa al archivo .keras
            path = os.path.join(root, fname)
            print(f"Cargando {fname} desde {root} …")
            model = load_model(path)
            
            # Evaluar el modelo
            evaluation_result = model.evaluate(X_eval, y_eval, verbose=0)
            
            # Si el resultado es una lista o tupla, extraemos las métricas
            if isinstance(evaluation_result, (list, tuple)):
                mse = evaluation_result[0]  # MSE (primer valor)
                mae = evaluation_result[1]  # MAE (segundo valor)
                rmse = np.sqrt(mse)        # RMSE (raíz cuadrada de MSE)
                mape = evaluation_result[2] if len(evaluation_result) > 2 else None  # MAPE (tercer valor si existe)
            else:
                mse = evaluation_result
                mae = rmse = mape = None
            
            # Añadir los resultados con las métricas
            results.append((mse, mae, rmse, mape, fname))
    
    # Ordenamos por MSE ascendente
    results.sort(key=lambda x: x[0])
    return results


def update_model_with_last_days(
    model_path: str,
    df: pd.DataFrame,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    days: int,
    lookback: int = 12,
    delay: int = 1,
    feature_cols: list = ['Open','High','Low','Close'],
    target_col: str = 'Target',
    epochs: int = 5,
    batch_size: int = 32,
    save_path: str = None
):
    """
    Fine-tuning del modelo con las últimas `days` días (o con el df ya filtrado si days=None).
    """
    df = df.sort_values('Datetime')
    if days is not None:
        cutoff = df['Datetime'].max() - pd.Timedelta(days=days)
        df_last = df[df['Datetime'] > cutoff].reset_index(drop=True)
    else:
        df_last = df.copy()

    # Crear ventanas
    Xn, yn = [], []
    n = len(df_last)
    for i in range(lookback, n - delay + 1):
        Xn.append(df_last[feature_cols].iloc[i-lookback:i].values)
        yn.append(df_last[target_col].iloc[i + delay - 1])
    Xn = np.array(Xn)
    yn = np.array(yn)

    if len(Xn) == 0:
        raise ValueError(f"No hay suficientes datos recientes para entrenar.")

    if np.isnan(yn).any():
        raise ValueError("La variable objetivo contiene valores NaN.")

    if (Xn <= 0).any() or (yn <= 0).any():
        raise ValueError("Valores no válidos (<=0) para aplicar log().")

    # Escalar
    X_flat = np.log(Xn).reshape(-1, len(feature_cols))
    Xn_scaled = scaler_X.transform(X_flat).reshape(Xn.shape)
    yn_log = np.log(yn).reshape(-1, 1)
    yn_scaled = scaler_y.transform(yn_log).flatten()

    # Cargar modelo y hacer fine-tuning
    model = load_model(model_path)
    model.fit(
        Xn_scaled, yn_scaled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # silencioso
    )

    # Guardar si se indica
    if save_path:
        model.save(save_path)

    return model


