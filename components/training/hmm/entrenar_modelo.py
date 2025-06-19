import json
import os
import sys
import argparse
import logging
import pandas as pd
from hmmlearn import hmm
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    parser.add_argument("--input", required=True, help="Ruta del archivo de entrada (.csv o .txt)")
    parser.add_argument("--output_model", required=True, help="Ruta donde guardar el modelo")
    parser.add_argument("--params", default="variables.json", help="Ruta al archivo JSON con parámetros del modelo")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        variables = json.load(f)

    claves_necesarias = ["columnas_input", "n_states", "covariance_type", "n_iter"]
    validar_json(variables, claves_necesarias)

    logging.info("Leyendo archivo de datos...")
    df = leer_datos(args.input)

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

    joblib.dump(model, args.output_model)
    logging.info(f"Modelo guardado en {args.output_model}")

if __name__ == "__main__":
    main()
