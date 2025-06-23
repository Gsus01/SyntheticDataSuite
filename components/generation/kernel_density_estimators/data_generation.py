
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_series_from_kde(kde_dict, n_series, series_length, start_id):
    # kdes_per_var_and_time es el diccionario {var: {t: kde}}
    kdes_per_var_and_time = kde_dict["models"]
    target_columns = kde_dict["target_columns"]

    generated = []
    for i in range(n_series):
        series_data = {col: [] for col in target_columns} # Cambiado a series_data para claridad
        series_data["ID"] = []
        series_data["Cycle_Index"] = []

        for t in range(1, series_length + 1):
            row_values = {}
            can_sample_all_vars_at_t = True

            # Iterar sobre cada variable objetivo para muestrear su valor en el tiempo 't'
            for var in target_columns:
                if var not in kdes_per_var_and_time or t not in kdes_per_var_and_time[var]:
                    logging.warning(f"No hay modelo KDE para la variable '{var}' en el paso {t}. Se omite esta fila para esta serie.")
                    can_sample_all_vars_at_t = False
                    break # Si falta un modelo para una variable en 't', no podemos generar esta fila

                kde_model = kdes_per_var_and_time[var][t]
                try:
                    sample_value = kde_model.sample(1)[0][0] # Extraer el valor escalar
                    row_values[var] = sample_value
                except ValueError as e:
                    logging.error(f"Error al muestrear KDE para la variable '{var}' en el paso {t}: {e}. Probablemente el modelo no fue entrenado correctamente (menos de 2 puntos). Se omite esta fila para esta serie.")
                    can_sample_all_vars_at_t = False
                    break

            if can_sample_all_vars_at_t:
                for col in target_columns:
                    series_data[col].append(row_values[col])
                series_data["ID"].append(start_id + i)
                series_data["Cycle_Index"].append(t)
            # Si no se pudo muestrear para todas las variables en 't', simplemente no añadimos esta fila.
            # Puedes decidir si prefieres rellenar con NaN o algún otro valor por defecto.


        # Asegurarse de que no estamos creando DataFrames vacíos si todas las filas se omitieron
        if any(series_data[col] for col in target_columns): # Verificar si alguna lista de valores tiene contenido
            df_series = pd.DataFrame(series_data)
            generated.append(df_series)
        else:
            logging.warning(f"No se pudieron generar datos válidos para la serie {start_id + i} en ninguna de sus posiciones. La serie se omite.")


    # Concatenar solo si hay DataFrames generados
    if generated:
        return pd.concat(generated, ignore_index=True)
    else:
        logging.warning("No se generaron series sintéticas válidas. Devolviendo DataFrame vacío.")
        return pd.DataFrame() # Devolver un DataFrame vacío si no se generó nada

def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas usando modelos KDE por timestep.")
    parser.add_argument("--model_path", required=True, help="Ruta al archivo .pkl del modelo KDE")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON: {key}")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"El archivo del modelo no se encontró en: {model_path}")
        exit(1)

    kde_dict = joblib.load(model_path)
    logging.info("Modelo KDE cargado correctamente.")

    df_generated = generate_series_from_kde(
        kde_dict,
        config["n_series"],
        config["series_length"],
        config["start_id"]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_generated.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
