
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

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

def generate_gp_series(model_dict, n_series, series_length, start_id):
    models = model_dict["models"]
    id_col = model_dict["id_col"]
    time_col = model_dict["time_col"]
    target_cols = model_dict["target_cols"]

    synthetic_data = []
    for i in range(n_series):
        time_steps = np.arange(1, series_length + 1).reshape(-1, 1)
        data = {time_col: time_steps.flatten(), id_col: [start_id + i] * series_length}
        for col in target_cols:
            gp = models[col]
            y_pred, _ = gp.predict(time_steps, return_std=True)
            data[col] = y_pred
        synthetic_data.append(pd.DataFrame(data))

    return pd.concat(synthetic_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Genera datos sintéticos con modelos GP multivariados.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el modelo entrenado")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar los datos sintéticos")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON de configuración")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config_path = discover_file(config_dir, ["*.json"], "de configuración JSON")
        model_path = discover_file(input_dir, ["*.pkl", "*.joblib"], "de modelo entrenado")
    except FileNotFoundError as exc:
        logging.error(exc)
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    logging.info(f"Configuración cargada desde {config_path}")

    required_keys = ["n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON.")
            exit(1)

    model_dict = joblib.load(model_path)
    logging.info(f"Modelo cargado correctamente desde {model_path}")

    df_synthetic = generate_gp_series(
        model_dict,
        config["n_series"],
        config["series_length"],
        config["start_id"]
    )

    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    df_synthetic.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos guardados en {output_path}")

if __name__ == "__main__":
    main()
