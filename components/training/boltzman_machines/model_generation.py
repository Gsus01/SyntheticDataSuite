
import pandas as pd
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo Restricted Boltzmann Machine (RBM).")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV con los datos")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar el modelo entrenado (.pkl)")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    # Cargar configuración
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Archivo de configuración no encontrado: {args.config}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = ["columns_to_use", "n_components", "learning_rate", "n_iter"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo de configuración JSON.")
            exit(1)

    # Leer datos
    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"Archivo de datos no encontrado: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    X = df[config["columns_to_use"]]

    logging.info(f"Entrenando RBM con columnas: {config['columns_to_use']}")

    # Preprocesamiento y modelo
    scaler = MinMaxScaler()
    rbm = BernoulliRBM(n_components=config["n_components"],
                       learning_rate=config["learning_rate"],
                       n_iter=config["n_iter"],
                       random_state=42)

    model = Pipeline(steps=[("scaler", scaler), ("rbm", rbm)])
    model.fit(X)

    # Guardar modelo
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "columns": config["columns_to_use"]
    }, output_path)

    logging.info(f"Modelo RBM guardado exitosamente en {output_path}")

if __name__ == "__main__":
    main()
