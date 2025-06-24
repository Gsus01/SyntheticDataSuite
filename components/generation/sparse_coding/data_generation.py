import numpy as np
import pandas as pd
import argparse
import json
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_sparse_series(dictionary, scaler, n_series, series_length, target_columns, start_id):
    synthetic_data = []
    n_features = dictionary.shape[1]
    for i in range(n_series):
        codes = np.random.laplace(loc=0.0, scale=1.0, size=(series_length, dictionary.shape[0]))
        data = np.dot(codes, dictionary)
        data = scaler.inverse_transform(data)

        df = pd.DataFrame(data, columns=target_columns)
        df["ID"] = start_id + i
        df["Cycle_Index"] = np.arange(1, series_length + 1)
        synthetic_data.append(df)

    return pd.concat(synthetic_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas usando codificación dispersa.")
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pkl entrenado con Sparse Coding")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"El modelo no se encontró en: {model_path}")
        exit(1)

    model = joblib.load(model_path)
    dictionary = model["dictionary"]
    scaler = model["scaler"]
    target_columns = model["target_columns"]

    df_generated = generate_sparse_series(
        dictionary,
        scaler,
        config["n_series"],
        config["series_length"],
        target_columns,
        config["start_id"]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_generated.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
