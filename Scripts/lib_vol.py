from preprocess import *
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

def log_return(df, column='Close'):
    log_returns = np.log(df[column] / df[column].shift(1)).dropna()
    return log_returns * (1e+4)

def test_stationarity(log_returns):
    result = adfuller(log_returns)
    print("Estadística ADF:", result[0])
    print("Valor p:", result[1])
    
    if result[1] < 0.05:
        print("Los datos son estacionarios.")
    else:
        print("Los datos NO son estacionarios.")


def plot_last_candles(df, num_velas=100):
    """
    Muestra un gráfico de los precios de cierre para las últimas 'num_velas' velas.
    """
    df_last = df[-num_velas:]
    plt.figure(figsize=(10, 6))
    plt.plot(df_last['Datetime'], df_last['Close'], label="Precio de Cierre", color='blue')
    plt.title(f"Últimas {num_velas} velas de EUR/USD")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_log_returns(df, num_velas=100, scale_factor=1e+4):
    """
    Esta función grafica los log-retornos escalados de las últimas 'num_velas' velas.
    """
    # Calcular log-retornos
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # Escalar los log-retornos
    log_returns_scaled = log_returns * scale_factor
    
    # Filtrar las últimas 'num_velas' velas
    log_returns_last = log_returns_scaled[-num_velas:]
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'][-num_velas:], log_returns_last, label="Log-Retornos Escalados", color='blue')
    plt.title(f"Últimos {num_velas} log-retornos escalados")
    plt.xlabel("Fecha")
    plt.ylabel("Log-Retornos Escalados")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(df, num_velas=100, window=30):
    """
    Muestra un gráfico de la volatilidad condicional calculada con una ventana deslizante.
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    rolling_volatility = log_returns.rolling(window=window).std()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'][-num_velas:], rolling_volatility[-num_velas:], label="Volatilidad (30 velas)", color='orange')
    plt.title(f"Volatilidad estimada para las últimas {num_velas} velas")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acf_returns(df, num_velas=100):
    """
    Muestra un gráfico de la autocorrelación de los log-retornos para las últimas 'num_velas' velas.
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    log_returns_last = log_returns[-num_velas:]
    
    plot_acf(log_returns_last, lags=40)
    plt.title(f"ACF de los log-retornos para las últimas {num_velas} velas")
    plt.show()


def plot_pacf_returns(df, num_velas=100):
    """
    Muestra un gráfico de la autocorrelación parcial de los log-retornos para las últimas 'num_velas' velas.
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    log_returns_last = log_returns[-num_velas:]
    
    plot_pacf(log_returns_last, lags=40)
    plt.title(f"PACF de los log-retornos para las últimas {num_velas} velas")
    plt.show()

def plot_log_returns_distribution(df, num_velas=100):
    """
    Muestra un histograma de la distribución de los log-retornos para las últimas 'num_velas' velas.
    """
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    log_returns_last = log_returns[-num_velas:]

    plt.figure(figsize=(10, 6))
    plt.hist(log_returns_last, bins=50, alpha=0.75, color='blue')
    plt.title(f"Distribución de los log-retornos para las últimas {num_velas} velas")
    plt.xlabel("Log-retornos")
    plt.ylabel("Frecuencia")
    plt.show()


#SE HACEN DOS PORQUE EN UNO MISMO NO SE PUEDE - DIFERENTE ESCALAS#
def plot_price_and_volatility_side_by_side(df, num_velas=5000, window=30):
    """
    Muestra un gráfico de los precios de cierre y la volatilidad escalada en dos subgráficos.
    """
    # Calcular log-retornos
    log_returns = log_return(df)

    # Calcular la volatilidad (desviación estándar de los log-retornos) en una ventana deslizante de 30 velas
    rolling_volatility = log_returns.rolling(window=window).std()

    # Escalar la volatilidad por el factor proporcionado
    scaled_volatility = rolling_volatility

    # Filtrar las últimas 'num_velas' velas
    df_last = df[-num_velas:]
    scaled_volatility_last = scaled_volatility[-num_velas:]

    # Crear los subgráficos
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de precios de cierre
    ax[0].plot(df_last['Datetime'], df_last['Close'], label="Precio de Cierre", color='blue')
    ax[0].set_title("Precio de Cierre")
    ax[0].set_xlabel("Fecha")
    ax[0].set_ylabel("Precio de Cierre")
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].legend()

    # Gráfico de volatilidad escalada
    ax[1].plot(df_last['Datetime'], scaled_volatility_last, label="Volatilidad Escalada (30 velas)", color='orange')
    ax[1].set_title("Volatilidad Escalada")
    ax[1].set_xlabel("Fecha")
    ax[1].set_ylabel("Volatilidad")
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].legend()

    # Ajustar y mostrar los subgráficos
    plt.tight_layout()
    plt.show()

def all_plots(df, num_velas=1000):
    """
    Esta función genera varios gráficos utilizando las funciones ya definidas para analizar la volatilidad y el precio.
    """
    # Plot 1: Últimas velas
    plot_last_candles(df, num_velas)
    
    # Plot 2: Volatilidad rollante
    plot_rolling_volatility(df, num_velas)
    
    # Plot 3: ACF de los retornos
    plot_acf_returns(df, num_velas)
    
    # Plot 4: PACF de los retornos
    plot_pacf_returns(df, num_velas)
    
    # Plot 5: Distribución de los log-retornos
    plot_log_returns_distribution(df, num_velas)
    
    # Plot 6: Precio y volatilidad al lado
    plot_price_and_volatility_side_by_side(df, num_velas, window=30)

def corr_vol_price(df, window=30): 
    """
    Calcula la correlación entre la volatilidad y el cambio de precio en una ventana deslizante de tamaño 'window'.
    """
    # Calcular log-retornos
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # Calcular la volatilidad en una ventana deslizante de 30 velas
    rolling_volatility = log_returns.rolling(window=window).std()

    # Calcular el cambio de precio (retornos simples)
    df['Price Change'] = df['Close'].pct_change()

    volatility_and_price_corr = df['Price Change'].rolling(window=window).corr(rolling_volatility)

    print("Correlación entre volatilidad y cambio de precio:", volatility_and_price_corr.tail(1))
    #Esta función te dará una visión clara de la relación entre la volatilidad y los movimientos de precio, 
    #lo cual es útil para estrategias de gestión de riesgo y ajuste de tamaño de posición.

def fit_garch_model(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp="off")  # Ajuste sin mostrar detalles
    return model_fit

def fit_gjr_garch_model(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q, o=1)  # 'o' especifica GJR-GARCH
    model_fit = model.fit(disp="off")
    return model_fit

def fit_garch_with_t_distribution(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='t')  # 'dist' especifica la distribución t
    model_fit = model.fit(disp="off")
    return model_fit

def fit_egarch_model(returns, p=1, q=1):
    model = arch_model(returns, vol='EGARCH', p=p, q=q)
    model_fit = model.fit(disp="off")
    return model_fit

def fit_arch_model(returns, q=1):
    model = arch_model(returns, vol='ARCH', p=1, q=q)  # p=0 indica que es un modelo ARCH
    model_fit = model.fit(disp="off")
    return model_fit

def fit_garch_m_model(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant')  # Volatilidad y media constante
    model_fit = model.fit(disp="off")
    return model_fit

def fit_aparch_model(returns, p=1, q=1):
    model = arch_model(returns, vol='APARCH', p=p, q=q)
    model_fit = model.fit(disp="off")
    return model_fit

def fit_figarch_model(returns, p=1, q=1, d=0.5):
    model = arch_model(returns, vol='FIGARCH', p=p, q=q)
    model.volatility.d = d  
    model_fit = model.fit(disp="off")
    return model_fit


def compare_garch_models(df, p_values=[1], q_values=[1]):
    # Calcular los log-retornos
    log_returns = log_return(df)

    results = []

    # Probar diferentes combinaciones de p y q
    for p in p_values:
        for q in q_values:
            # Ajustar el modelo GARCH
            model = arch_model(log_returns, vol='Garch', p=p, q=q)
            model_fit = model.fit(disp="off")
            
            # Guardar los resultados
            results.append({
                'p': p,
                'q': q,
                'AIC': model_fit.aic,
                'BIC': model_fit.bic,
                'Log-Likelihood': model_fit.loglikelihood
            })
    
    # Convertir los resultados en un DataFrame para  fácil comparación
    results_df = pd.DataFrame(results)
    return results_df

def plot_model_summaries(df, p, q, d=0.5):
    log_returns = log_return(df)

    # Definir los modelos
    models = {
        "GARCH": fit_garch_model,
        "GARCH-M": fit_garch_m_model,
        "GJR-GARCH": fit_gjr_garch_model,
        "APARCH": fit_aparch_model,
        "FIGARCH": fit_figarch_model,
        "EGARCH": fit_egarch_model,
        "ARCH": fit_arch_model
    }

    # Entrenar y mostrar los summaries para cada modelo
    for model_name, model_func in models.items():
        print(f"\n\nFitting {model_name}...")
        
        # Llamar a la función correspondiente para cada modelo y ajustar
        if model_name == "FIGARCH":
            model_fit = model_func(log_returns, p=1, q=1, d=d)
        elif model_name == "ARCH":
            model_fit = model_func(log_returns, q=q)
        else:
            model_fit = model_func(log_returns, p=p, q=q)
        
        # Imprimir el summary del modelo ajustado
        print(model_fit.summary())

def compare_figarch_models(df, p_values=[1, 2, 3], q_values=[1, 2, 3], d_values=[0.5]):
    # Calcular log-retornos
    log_returns = log_return(df)

    # Crear una lista para almacenar los resultados
    results = []

    # Probar diferentes combinaciones de p, q, y d
    for p in p_values:
        for q in q_values:
            for d in d_values:
                print(f"\nFitting FIGARCH(p={p}, q={q}, d={d})...")
                model_fit = fit_figarch_model(log_returns, p=p, q=q, d=d)
                results.append({
                    'p': p,
                    'q': q,
                    'd': d,
                    'AIC': model_fit.aic,
                    'BIC': model_fit.bic,
                    'Log-Likelihood': model_fit.loglikelihood
                })

    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def predict_volatility(model_fit, horizon=1):
    # Predicción de varianza futura (horizon pasos)
    forecast = model_fit.forecast(horizon=horizon)
    
    # Extraer la varianza predicha (última varianza predicha en el horizonte)
    predicted_variance = forecast.variance.values[-1, :]
    
    # Tomar la raíz cuadrada de la varianza para obtener la volatilidad
    predicted_volatility = np.sqrt(predicted_variance)
    
    return predicted_volatility


def calculate_metrics(model_fit, real_data, horizon=1):
    """
    Calcula RMSE, MSE y MAPE para el modelo ajustado.
    """
    # Predecir la volatilidad futura utilizando el modelo ajustado
    predicted_volatility = np.sqrt(model_fit.forecast(horizon=horizon).variance.values[-1, :])
    
    # Asegurar que ambas series tengan la misma longitud
    min_len = min(len(real_data), len(predicted_volatility))
    real_data = real_data[-min_len:]
    predicted_volatility = predicted_volatility[:min_len]
    
    # Calcular MSE (Mean Squared Error)
    mse = mean_squared_error(real_data, predicted_volatility)
    
    # Calcular RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((real_data - predicted_volatility) / real_data)) * 100
    
    # Retornar las métricas como un diccionario
    return {
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape
    }

def plot_model_metrics(df, p, q, d=0.5):
    """
    Ajusta varios modelos GARCH, GARCH-M, GJR-GARCH, APARCH, FIGARCH, y EGARCH a los datos y 
    muestra las métricas de rendimiento (RMSE, MSE, MAPE) para cada uno.
    """
    # Calcular log-retornos
    log_returns = log_return(df)

    # Definir los modelos
    models = {
        "GARCH": fit_garch_model,
        "GARCH-M": fit_garch_m_model,
        "GJR-GARCH": fit_gjr_garch_model,
        "APARCH": fit_aparch_model,
        "FIGARCH": fit_figarch_model,
        "EGARCH": fit_egarch_model,
        "ARCH": fit_arch_model
    }

    # Crear una lista para almacenar los resultados
    results = []

    # Entrenar y mostrar las métricas para cada modelo
    for model_name, model_func in models.items():
        print(f"\n\nFitting {model_name}...")
        
        # Llamar a la función correspondiente para cada modelo y ajustar
        if model_name == "FIGARCH":
            model_fit = model_func(log_returns, p=1, q=1, d=d)
        elif model_name == "ARCH":
            model_fit = model_func(log_returns, q=q)
        else:
            model_fit = model_func(log_returns, p=p, q=q)
        
        # Calcular las métricas
        real_volatility = log_returns.rolling(window=30).std().dropna()  # Volatilidad real para comparación
        metrics = calculate_metrics(model_fit, real_volatility, horizon=1)
        
        # Almacenar los resultados
        results.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MSE': metrics['MSE'],
            'MAPE': metrics['MAPE']
        })
    
    # Convertir los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    print("\nMétricas de los modelos:")
    print(results_df)