
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    parser.add_argument("--model_path", required=True, help="Ruta al archivo .pkl del modelo")
    parser.add_argument("--output_path", required=True, help="Ruta de salida para los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON.")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"Archivo del modelo no encontrado en: {model_path}")
        exit(1)

    model_dict = joblib.load(model_path)
    logging.info("Modelo cargado correctamente.")

    df_synthetic = generate_gp_series(
        model_dict,
        config["n_series"],
        config["series_length"],
        config["start_id"]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos guardados en {output_path}")

if __name__ == "__main__":
    main()
