
import pandas as pd
import argparse
import json
import joblib
import logging
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def entrenar_modelo_serie(serie, model_type, arima_order, sarima_order):
    if model_type == "ARIMA":
        model = ARIMA(serie, order=arima_order)
    elif model_type == "SARIMA":
        p,d,q,P,D,Q,s = sarima_order
        model = SARIMAX(serie, order=(p,d,q), seasonal_order=(P,D,Q,s))
    else:
        raise ValueError("Tipo de modelo no soportado. Usa 'ARIMA' o 'SARIMA'")
    return model.fit()

def main():
    parser = argparse.ArgumentParser(description="Entrena modelos ARIMA o SARIMA por ID y variable.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output_path", required=True, help="Ruta para guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["model_type", "variables", "ids"]
    for k in required_keys:
        if k not in config:
            logging.error(f"Falta la clave requerida en config: {k}")
            exit(1)

    model_type = config["model_type"]
    variables = config["variables"]
    ids = config["ids"]
    arima_order = tuple(config.get("arima_order", [1,1,1]))
    sarima_order = tuple(config.get("sarima_order", [1,1,1,1,1,1,12]))

    df = pd.read_csv(args.input_path)
    logging.info(f"Datos cargados: {df.shape}")

    modelos = {}
    for id_ in ids:
        for var in variables:
            serie = df[df["ID"] == id_][var].dropna()
            logging.info(f"Entrenando modelo {model_type} para ID={id_}, variable='{var}'")
            try:
                modelo_entrenado = entrenar_modelo_serie(serie, model_type, arima_order, sarima_order)
                modelos[(id_, var)] = modelo_entrenado
            except Exception as e:
                logging.warning(f"Error entrenando modelo para ID={id_}, var={var}: {e}")

    # Ensure output directory exists:
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Directorio creado: {output_dir}")
    joblib.dump(modelos, args.output_path)
    logging.info(f"Modelos guardados en {args.output_path}")

if __name__ == "__main__":
    main()
