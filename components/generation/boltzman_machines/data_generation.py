
import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    parser.add_argument("--model_path", required=True, help="Path to the trained RBM model (.pkl)")
    parser.add_argument("--output_path", required=True, help="Path to save the generated synthetic data (.csv)")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {args.config}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    n_series = config.get("n_series")
    series_length = config.get("series_length")

    if n_series is None or series_length is None:
        logging.error("Missing 'n_series' or 'series_length' in config JSON.")
        exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"Model not found at: {model_path}")
        exit(1)

    loaded_model_data = joblib.load(model_path)
    model_pipeline = loaded_model_data["model"]
    columns_to_use = loaded_model_data["columns"]
    logging.info("RBM Model (Pipeline) loaded successfully.")

    synthetic_df = generate_synthetic_data(model_pipeline, columns_to_use, n_series, series_length)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)
    logging.info(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    main()