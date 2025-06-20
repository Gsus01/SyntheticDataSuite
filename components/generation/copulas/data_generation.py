import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos a partir de un modelo de cópulas multivariadas.")
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pkl entrenado")
    parser.add_argument("--output_data_path", required=True, help="Ruta donde guardar los datos sintéticos generados")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    args = parser.parse_args()

    # Cargar configuración
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Archivo de configuración no encontrado: {args.config}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = ["n_samples"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    n_samples = config["n_samples"]

    # Verificar que el modelo exista
    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"El modelo no se encontró en: {model_path}")
        exit(1)

    # Cargar modelo
    model = joblib.load(model_path)
    logging.info("Modelo de cópulas cargado exitosamente.")

    # Generar datos sintéticos
    logging.info(f"Generando {n_samples} muestras sintéticas...")
    samples = model.sample(n_samples)

    # Guardar
    output_path = Path(args.output_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos guardados en {output_path}")

if __name__ == "__main__":
    main()
