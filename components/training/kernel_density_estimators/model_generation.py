
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.neighbors import KernelDensity

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_kde_per_position(df, id_col, time_col, target_cols, max_length):
    kde_models = {var: {} for var in target_cols}

    for var in target_cols:
        logging.info(f"Entrenando KDEs para variable: {var}")
        for t in range(1, max_length + 1):
            values = df[df[time_col] == t][var].dropna().values.reshape(-1, 1)
            if len(values) > 1:
                kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
                kde.fit(values)
                kde_models[var][t] = kde
            else:
                logging.warning(f"No hay suficientes valores para {var} en t={t}, se omite.")
    return kde_models

def main():
    parser = argparse.ArgumentParser(description="Entrena modelos KDE por posición temporal.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV con series temporales")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar el modelo KDE (.pkl)")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    # Cargar configuración
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"No se encontró el archivo de configuración: {args.config}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = ["id_column", "time_column", "target_columns", "max_length"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave '{key}' en el JSON de configuración.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]
    max_length = config["max_length"]

    # Cargar datos
    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"Archivo de entrada no encontrado: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    logging.info(f"Datos cargados desde {args.input_path}, columnas objetivo: {target_cols}")

    kde_models = train_kde_per_position(df, id_col, time_col, target_cols, max_length)

    # Guardar el modelo completo
    joblib.dump({
        "models": kde_models,
        "target_columns": target_cols,
        "time_column": time_col,
        "max_length": max_length
    }, args.output_path)

    logging.info(f"Modelos KDE guardados exitosamente en {args.output_path}")

if __name__ == "__main__":
    main()
