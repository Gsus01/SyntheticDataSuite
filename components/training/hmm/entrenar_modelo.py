import json
import os
import sys
import argparse
import logging
import pandas as pd
from hmmlearn import hmm
import joblib
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

def validar_json(variables, claves_requeridas):
    faltantes = [k for k in claves_requeridas if k not in variables]
    if faltantes:
        logging.error(f"Faltan claves necesarias en el JSON: {faltantes}")
        sys.exit(1)

def leer_datos(path):
    if not os.path.isfile(path):
        logging.error(f"El archivo {path} no existe.")
        sys.exit(1)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".txt"):
        return pd.read_csv(path, delim_whitespace=True, header=None)
    else:
        logging.error("Formato no compatible. Usa .csv o .txt")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Entrenar un modelo HMM")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el archivo de entrada (.csv o .txt)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar el modelo")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON con parámetros del modelo")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        params_path = discover_file(config_dir, ["*.json"], "de parámetros JSON")
        input_path = discover_file(input_dir, ["*.csv", "*.txt"], "de datos de entrada")
    except FileNotFoundError as exc:
        logging.error(exc)
        sys.exit(1)

    with open(params_path, "r") as f:
        variables = json.load(f)

    claves_necesarias = ["columnas_input", "n_states", "covariance_type", "n_iter"]
    validar_json(variables, claves_necesarias)

    logging.info("Leyendo archivo de datos...")
    df = leer_datos(str(input_path))

    indices = variables["columnas_input"]
    columnas = [df.columns[i] for i in indices]
    data = df[columnas].values

    logging.info("Entrenando modelo HMM...")
    model = hmm.GaussianHMM(
        n_components=variables["n_states"],
        covariance_type=variables["covariance_type"],
        n_iter=variables["n_iter"]
    )
    model.fit(data)

    output_path = output_dir / "hmm_model.pkl"
    joblib.dump(model, output_path)
    logging.info(f"Modelo guardado en {output_path}")

if __name__ == "__main__":
    main()
