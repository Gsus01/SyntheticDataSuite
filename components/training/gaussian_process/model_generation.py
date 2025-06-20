
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output_model_path", required=True, help="Ruta para guardar el modelo entrenado (.pkl)")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con la configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["id_column", "time_column", "target_columns"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el JSON de configuración.")
            exit(1)

    id_col = config["id_column"]
    time_col = config["time_column"]
    target_cols = config["target_columns"]

    df = pd.read_csv(args.input_path)
    logging.info(f"Datos cargados desde {args.input_path}, columnas objetivo: {target_cols}")

    models = train_gp_models(df, id_col, time_col, target_cols)

    joblib.dump({
        "models": models,
        "id_col": id_col,
        "time_col": time_col,
        "target_cols": target_cols
    }, args.output_model_path)

    logging.info(f"Modelos guardados en {args.output_model_path}")

if __name__ == "__main__":
    main()
