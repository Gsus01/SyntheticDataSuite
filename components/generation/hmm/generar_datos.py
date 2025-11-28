import json
import sys
import argparse
import logging
import pandas as pd
import joblib
import numpy as np
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

def main():
    parser = argparse.ArgumentParser(description="Generar datos sintéticos usando un modelo HMM entrenado")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio con el modelo entrenado (.pkl/.joblib)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio para guardar los datos sintéticos")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON de parámetros")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        params_path = discover_file(config_dir, ["*.json"], "de parámetros JSON")
        model_path = discover_file(input_dir, ["*.pkl", "*.joblib"], "de modelo HMM")
    except FileNotFoundError as exc:
        logging.error(exc)
        sys.exit(1)

    with open(params_path, "r") as f:
        variables = json.load(f)

    claves_necesarias = ["n_series", "longitud", "columnas_input"]
    validar_json(variables, claves_necesarias)

    model = joblib.load(model_path)
    logging.info(f"Modelo HMM cargado desde {model_path}")

    logging.info("Generando series sintéticas...")
    series = []
    for _ in range(variables["n_series"]):
        X, _ = model.sample(variables["longitud"])
        series.append(X)

    synthetic = np.stack(series)
    df_out = pd.DataFrame(
        synthetic.reshape(-1, synthetic.shape[-1]),
        columns=[f"feature_{i}" for i in range(synthetic.shape[-1])]
    )
    
    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    df_out.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos guardados en {output_path}")

if __name__ == "__main__":
    main()
