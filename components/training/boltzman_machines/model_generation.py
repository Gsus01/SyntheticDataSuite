
import pandas as pd
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
    parser = argparse.ArgumentParser(description="Entrena un modelo Restricted Boltzmann Machine (RBM).")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el CSV con los datos")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar el modelo entrenado (.pkl)")
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

    required_keys = ["columns_to_use", "n_components", "learning_rate", "n_iter"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo de configuración JSON.")
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
    output_path = output_dir / "rbm_model.pkl"
    joblib.dump({
        "model": model,
        "columns": config["columns_to_use"]
    }, output_path)

    logging.info(f"Modelo RBM guardado exitosamente en {output_path}")

if __name__ == "__main__":
    main()
