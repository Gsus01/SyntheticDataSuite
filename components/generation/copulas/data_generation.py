import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_INPUT_DIR = "/data/inputs"
DEFAULT_OUTPUT_DIR = "/data/outputs"
DEFAULT_CONFIG_DIR = "/data/config"


def discover_file(directory: Path, patterns, description: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado para {description}: {directory}")
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontraron archivos {description} en {directory} (patrones: {', '.join(patterns)})")

def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos a partir de un modelo de cópulas multivariadas.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el modelo entrenado (.pkl)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar los datos sintéticos generados")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo de configuración JSON.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config_path = discover_file(config_dir, ["*.json"], "de configuración JSON")
        model_path = discover_file(input_dir, ["*.pkl", "*.joblib"], "de modelo .pkl")
    except FileNotFoundError as exc:
        logging.error(exc)
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    logging.info(f"Configuración cargada desde {config_path}")

    required_keys = ["n_samples"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    n_samples = config["n_samples"]

    # Cargar modelo
    model = joblib.load(model_path)
    logging.info(f"Modelo de cópulas cargado desde {model_path}")

    # Generar datos sintéticos
    logging.info(f"Generando {n_samples} muestras sintéticas...")
    samples = model.sample(n_samples)

    # Guardar
    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    samples.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos guardados en {output_path}")

if __name__ == "__main__":
    main()
