
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de codificación dispersa (Sparse Coding) multivariado.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output_path", required=True, help="Ruta para guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["id_column", "time_column", "target_columns", "n_components", "alpha", "max_iter"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]

    df = pd.read_csv(args.input_path)
    logging.info(f"Archivo cargado con {len(df)} registros.")

    # Agrupar por ID y concatenar todas las series (en línea)
    grouped = df.groupby(id_col)
    series_data = []
    for _, group in grouped:
        group_sorted = group.sort_values(by=time_col)
        values = group_sorted[target_cols].values
        series_data.append(values)
    full_data = np.vstack(series_data)

    scaler = StandardScaler()
    full_data_scaled = scaler.fit_transform(full_data)

    logging.info("Entrenando modelo DictionaryLearning (sparse coding)...")
    dict_learner = DictionaryLearning(
        n_components=config["n_components"],
        alpha=config["alpha"],
        max_iter=config["max_iter"],
        random_state=42
    )
    codes = dict_learner.fit_transform(full_data_scaled)
    dictionary = dict_learner.components_

    # Guardar el modelo
    joblib.dump({
        "dictionary": dictionary,
        "scaler": scaler,
        "target_columns": target_cols
    }, args.output_path)

    logging.info(f"Modelo guardado exitosamente en {args.output_path}")

if __name__ == "__main__":
    main()
