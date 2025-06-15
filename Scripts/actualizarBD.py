import yfinance as yf
import pandas as pd

def actualizaBD():
    # 1) DESCARGA DE DATOS DESDE YFINANCE
    archivo_salida = "./data/temp.xlsx"
    print("Actualizando BD...")
    datos = yf.download("EURUSD=X", period="60d", interval="5m")
    if datos.empty:
        print("Error: Fallo al coger los datos de yfinance.")
        return
    datos.index = datos.index.tz_localize(None)
    datos.to_excel(archivo_salida)

    # 2) CONVERSIÓN DEL FORMATO
    df = pd.read_excel(archivo_salida, skiprows=2)
    df.columns = ["Datetime", "Close", "High", "Low", "Open", "_Volume"]
    df = df[["Datetime", "Open", "High", "Low", "Close"]]  # Quitamos la columna Volume
    df.to_excel(archivo_salida, index=False)

    # 3) UNIÓN CON LA BASE EXISTENTE
    ruta_eurusd_5m = "./data/eurusd_5m_final.xlsx"  
    df_original = pd.read_excel(ruta_eurusd_5m)
    if "Volume" in df_original.columns:
        df_original.drop(columns=["Volume"], inplace=True)

    df_nuevo = pd.read_excel(archivo_salida)
    min_nueva_fecha = df_nuevo["Datetime"].min()
    df_original = df_original[df_original["Datetime"] < min_nueva_fecha]
    df_unido = pd.concat([df_original, df_nuevo], ignore_index=True)
    df_unido.drop_duplicates(subset=["Datetime"], keep="last", inplace=True)
    df_unido.sort_values("Datetime", inplace=True)

    # 4) GUARDAR LA BD FINAL ACTUALIZADA 
    df_unido.to_excel(ruta_eurusd_5m, index=False)
    print(f"Base de datos actualizada")