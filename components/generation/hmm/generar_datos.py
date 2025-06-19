import json
import os
import sys
import argparse
import logging
import pandas as pd
import joblib
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def validar_json(variables, claves_requeridas):
    faltantes = [k for k in claves_requeridas if k not in variables]
    if faltantes:
        logging.error(f"Faltan claves necesarias en el JSON: {faltantes}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generar datos sintéticos usando un modelo HMM entrenado")
    parser.add_argument("--model", required=True, help="Ruta al modelo entrenado (joblib/pkl)")
    parser.add_argument("--output_data", required=True, help="Ruta para guardar los datos sintéticos")
    parser.add_argument("--params", default="variables.json", help="Ruta al archivo JSON con parámetros")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        variables = json.load(f)

    claves_necesarias = ["n_series", "longitud", "columnas_input"]
    validar_json(variables, claves_necesarias)

    model = joblib.load(args.model)

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
    df_out.to_csv(args.output_data, index=False)
    logging.info(f"Datos sintéticos guardados en {args.output_data}")

if __name__ == "__main__":
    main()
