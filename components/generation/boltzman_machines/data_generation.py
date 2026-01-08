
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def discover_file(directory: Path, patterns, description: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado para {description}: {directory}")
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontraron archivos {description} en {directory} (patrones: {', '.join(patterns)})")

def generate_synthetic_data(model_pipeline, columns_to_use, n_series, series_length):
    """
    Generates synthetic data from a trained RBM model (Pipeline).
    Note: sklearn's BernoulliRBM is primarily for feature extraction.
    This generation method provides scaled-inverse random data, not true RBM Gibbs sampling.
    For true RBM generative capabilities, custom implementation or other libraries are needed.
    """
    all_synthetic_data = []
    logging.info(f"Generating {n_series} synthetic series of length {series_length}...")

    scaler = model_pipeline.named_steps['scaler']
    
    for i in range(n_series):
        # Generate random data in the [0, 1] scaled space
        synthetic_scaled_data = np.random.rand(series_length, len(columns_to_use))
        
        # Inverse transform to original data scale
        synthetic_original_scale_data = scaler.inverse_transform(synthetic_scaled_data)
        
        series_df = pd.DataFrame(synthetic_original_scale_data, columns=columns_to_use)
        series_df['series_id'] = i + 1
        series_df['time_index'] = np.arange(series_length) + 1
        
        all_synthetic_data.append(series_df)

    return pd.concat(all_synthetic_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data from a trained RBM model.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el modelo entrenado")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar los datos sintéticos")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo de configuración JSON")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
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

    n_series = config.get("n_series")
    series_length = config.get("series_length")

    if n_series is None or series_length is None:
        logging.error("Missing 'n_series' or 'series_length' in config JSON.")
        exit(1)

    loaded_model_data = joblib.load(model_path)
    model_pipeline = loaded_model_data["model"]
    columns_to_use = loaded_model_data["columns"]
    logging.info(f"Modelo RBM cargado desde {model_path}")

    synthetic_df = generate_synthetic_data(model_pipeline, columns_to_use, n_series, series_length)

    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    synthetic_df.to_csv(output_path, index=False)
    logging.info(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    main()
