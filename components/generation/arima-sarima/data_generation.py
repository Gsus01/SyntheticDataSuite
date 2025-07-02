
import argparse
import json
import joblib
import pandas as pd
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Genera series futuras usando modelos ARIMA/SARIMA entrenados.")
    parser.add_argument("--model_path", required=True, help="Ruta al archivo .pkl con los modelos")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar las predicciones")
    parser.add_argument("--config", required=True, help="Archivo JSON con 'steps' (pasos a predecir)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if "steps" not in config:
        logging.error("El archivo JSON debe contener el campo 'steps'")
        exit(1)

    steps = config["steps"]
    modelos = joblib.load(args.model_path)
    logging.info(f"Modelos cargados: {len(modelos)}")

    all_preds = []
    for (id_, var), modelo in modelos.items():
        try:
            pred = modelo.forecast(steps=steps)
            df_pred = pd.DataFrame({
                "ID": id_,
                "Variable": var,
                "Step": range(1, steps + 1),
                "Prediction": pred
            })
            all_preds.append(df_pred)
        except Exception as e:
            logging.warning(f"Error generando predicción para ID={id_}, var={var}: {e}")

    df_all = pd.concat(all_preds, ignore_index=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)
    logging.info(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    main()
