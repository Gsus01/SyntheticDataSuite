import numpy as np
import pandas as pd
import argparse
import json
import joblib
import logging
from pathlib import Path

DEFAULT_INPUT_DIR = "/data/inputs"
DEFAULT_OUTPUT_DIR = "/data/outputs"
DEFAULT_CONFIG_DIR = "/data/config"

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def discover_file(directory: Path, patterns, description: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado para {description}: {directory}")
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontraron archivos {description} en {directory} (patrones: {', '.join(patterns)})")

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
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el modelo entrenado (.pkl)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar los datos sintéticos")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON de configuración")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config_path = discover_file(config_dir, ["*.json"], "de configuración JSON")
        model_path = discover_file(input_dir, ["*.pkl", "*.joblib"], "de modelo entrenado")
    except FileNotFoundError as exc:
        logging.error(exc)
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    logging.info(f"Configuración cargada desde {config_path}")

    required_keys = ["n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    model = joblib.load(model_path)
    logging.info(f"Modelo de codificación dispersa cargado desde {model_path}")
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

    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    df_generated.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
