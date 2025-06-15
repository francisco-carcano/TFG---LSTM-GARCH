from lib_ml import *
import numpy as np
from actualizarBD import actualizaBD
import pandas as pd
from tensorflow.keras import Input
from datetime import timedelta

#actualizaBD()

# Cargar y preparar los datos
path = "./Data/eurusd_5m_final.xlsx"

# 1) Carga y target de regresión
df = read_df(path, False)
df = target(df)

# Separar las dos últimas semanas
df['Datetime'] = pd.to_datetime(df['Datetime'])  # Asegurarse del tipo
last_date = df['Datetime'].max()
cutoff_date = last_date - timedelta(days=14)

df_recent = df[df['Datetime'] > cutoff_date].copy()      # Últimas 2 semanas
df_train = df[df['Datetime'] <= cutoff_date].copy()      # Todo lo anterior

# 2) Ventanas y escalado log+MinMax (solo con df_train)
X, y = windows(df_train, lookback=12, delay=1,
               feature_cols=['Open','High','Low','Close'],
               target_col='Target')
X_scaled, y_scaled, scaler_X, scaler_y = escalar_datos(X, y)

# 3) Train/Val/Test
X_tr, y_tr, X_val, y_val, X_te, y_te = split_data(X_scaled, y_scaled, 0.8, 0.2)

# Combinación 5: Bidirectional + Dense
train_and_save_models(
    X_tr, y_tr, X_val, y_val,
    lookback=12, n_features=X.shape[2],
    units_list=[256],
    lr_list=[0.0002],
    optimizer_list=['sgd'],
    epochs=50,
    batch_size=32,
    save_dir='./models/LSTM/Final',
    use_stacked=True,
    use_dense=False,
    use_bidirectional=True
)

train_and_save_models(
    X_tr, y_tr, X_val, y_val,
    lookback=12, n_features=X.shape[2],
    units_list=[256],
    lr_list=[0.0002],
    optimizer_list=['sgd'],
    epochs=50,
    batch_size=32,
    save_dir='./models/LSTM/Final',
    use_stacked=False,
    use_dense=False,
    use_bidirectional=True
)

###ENTRENARLO DEJANDO LAS DOS ULTIMAS SEMANAS DE DATOS SIN PASARLE PARA EL BACKTESTING###


# train_and_save_models(
#     X_tr, y_tr, X_val, y_val,
#     lookback=12, n_features=X.shape[2],
#     units_list=[256],
#     lr_list=[0.0002],
#     optimizer_list=['sgd'],
#     epochs=100,
#     batch_size=32,
#     save_dir='./models/LSTM/ConCapas',
#     use_stacked=True,
#     use_dense=True,
#     use_bidirectional=True
# )