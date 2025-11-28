
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.neighbors import KernelDensity

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
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el archivo CSV con series temporales")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar el modelo KDE (.pkl)")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON de configuración")
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

    required_keys = ["id_column", "time_column", "target_columns", "max_length"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave '{key}' en el JSON de configuración.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]
    max_length = config["max_length"]

    df = pd.read_csv(input_path)
    logging.info(f"Datos cargados desde {input_path}, columnas objetivo: {target_cols}")

    kde_models = train_kde_per_position(df, id_col, time_col, target_cols, max_length)

    # Guardar el modelo completo
    output_path = output_dir / "kde_model.pkl"
    joblib.dump({
        "models": kde_models,
        "target_columns": target_cols,
        "time_column": time_col,
        "max_length": max_length
    }, output_path)

    logging.info(f"Modelos KDE guardados exitosamente en {output_path}")

if __name__ == "__main__":
    main()
