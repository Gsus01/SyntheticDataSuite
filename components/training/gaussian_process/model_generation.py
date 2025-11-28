
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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

def train_gp_models(df, id_col, time_col, target_cols):
    models = {}
    grouped = df.groupby(id_col)
    for col in target_cols:
        X_all, y_all = [], []
        for _, group in grouped:
            X = group[time_col].values.reshape(-1, 1)
            y = group[col].values
            X_all.append(X)
            y_all.append(y)
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        gp.fit(X_all, y_all)
        models[col] = gp
        logging.info(f"Modelo GP entrenado para la variable: {col}")
    return models

def main():
    parser = argparse.ArgumentParser(description="Entrena modelos Gaussian Process para series multivariadas.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el archivo CSV de entrada")
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

    required_keys = ["id_column", "time_column", "target_columns"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON de configuración.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]

    df = pd.read_csv(input_path)
    logging.info(f"Datos cargados desde {input_path}, columnas objetivo: {target_cols}")

    models = train_gp_models(df, id_col, time_col, target_cols)

    output_path = output_dir / "gaussian_process_model.pkl"
    joblib.dump({
        "models": models,
        "id_col": id_col,
        "time_col": time_col,
        "target_cols": target_cols
    }, output_path)

    logging.info(f"Modelos guardados en {output_path}")

if __name__ == "__main__":
    main()
