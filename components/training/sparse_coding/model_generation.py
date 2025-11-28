
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler

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

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de codificación dispersa (Sparse Coding) multivariado.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el archivo CSV de entrada")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio para guardar el modelo")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON con configuración")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config_path = discover_file(config_dir, ["*.json"], "de configuración JSON")
        input_path = discover_file(input_dir, ["*.csv"], "de datos de entrada")
    except FileNotFoundError as exc:
        logging.error(exc)
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    logging.info(f"Configuración cargada desde {config_path}")

    required_keys = ["id_column", "time_column", "target_columns", "n_components", "alpha", "max_iter"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]

    df = pd.read_csv(input_path)
    logging.info(f"Archivo cargado con {len(df)} registros desde {input_path}.")

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
    output_path = output_dir / "sparse_coding_model.pkl"
    joblib.dump({
        "dictionary": dictionary,
        "scaler": scaler,
        "target_columns": target_cols
    }, output_path)

    logging.info(f"Modelo guardado exitosamente en {output_path}")

if __name__ == "__main__":
    main()
