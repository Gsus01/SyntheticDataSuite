import pandas as pd
import json
import argparse
import joblib
import logging
from pathlib import Path
from copulas.multivariate import GaussianMultivariate

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de cópulas multivariadas.")
    parser.add_argument("--input_path", required=True, help="Ruta del archivo de entrada (.csv o .txt)")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    args = parser.parse_args()

    # Leer configuración
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Archivo de configuración no encontrado: {args.config}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = ["columns_to_use"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    columns_to_use = config["columns_to_use"]

    if not input_path.exists():
        logging.error(f"Archivo de datos no encontrado: {input_path}")
        exit(1)

    # Cargar y preparar datos
    df = pd.read_csv(input_path)
    df = df[columns_to_use]
    logging.info(f"Datos cargados con columnas: {columns_to_use}")

    # Entrenar modelo
    model = GaussianMultivariate()
    model.fit(df)
    logging.info("Modelo de cópulas entrenado correctamente.")

    # Guardar modelo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logging.info(f"Modelo guardado en {output_path}")

if __name__ == "__main__":
    main()
