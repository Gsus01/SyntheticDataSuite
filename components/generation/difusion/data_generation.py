import pandas as pd
import numpy as np 
import json
import argparse
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader 
import os 
import _pickle 

# Define the DiffusionModel class - IT MUST BE IDENTICAL TO THE ONE USED FOR TRAINING
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, max_len, model_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_encoding[:x.size(1), :].unsqueeze(0)
        x = self.transformer(x)
        return self.output(x)

# Proceso de difusión (should be part of the model or a separate utility)
def diffusion_process(x, t, noise_schedule):
    alpha = 1 - noise_schedule[t]
    alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # Expandir dimensiones
    noise = torch.randn_like(x)
    return torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise, noise 

# Generation function
def generate_samples(model, noise_schedule, T, num_samples, seq_len, num_features):
    model.eval() 
    samples = torch.randn(num_samples, seq_len, num_features) 

    for t_step in reversed(range(T)):
        t = torch.tensor([t_step], dtype=torch.long)
        
        alpha_t = 1 - noise_schedule[t_step]
        alpha_t_exp = alpha_t.unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            predicted_noise = model(samples)

        samples = (samples - (1 - alpha_t_exp) * predicted_noise) / torch.sqrt(alpha_t_exp)
        
    return samples

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas usando un modelo de difusión.")
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pkl entrenado con Sparse Coding")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required_keys = ["n_series"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo JSON.")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"Modelo no encontrado: {model_path}")
        exit(1)

    logging.info(f"Cargando modelo y escalador desde {model_path}...")
    
    # Hemos intentado esto, pero el error persiste.
    # torch.serialization.add_safe_globals([
    #     StandardScaler,
    #     np.array(0).dtype.type
    # ])

    try:
        # --- CAMBIO AQUÍ: Forzar weights_only=False ---
        logging.warning("Intentando cargar con weights_only=False. Esto anula las comprobaciones de seguridad. Úsalo solo si confías en el origen del archivo.")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False) 
    except _pickle.UnpicklingError as e:
        logging.error(f"Error final al cargar el checkpoint incluso con weights_only=False: {e}")
        logging.error("Esto es inusual. El archivo del modelo podría estar corrupto o ser incompatible con la versión actual de PyTorch/Python.")
        exit(1)


    # Extract components from the loaded checkpoint
    scaler = checkpoint["scaler"]
    features = checkpoint["features"]
    max_len = checkpoint["max_len"]
    T = checkpoint["T"] # Number of diffusion steps
    
    # Recreate the model structure first
    model = DiffusionModel(
        input_dim=len(features), 
        max_len=max_len,         
        model_dim=64,            
        num_heads=4,             
        num_layers=3             
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 

    noise_schedule = torch.linspace(1e-4, 0.02, T) 

    logging.info("Generando series sintéticas...")
    
    generated_scaled_samples = generate_samples(
        model, 
        noise_schedule, 
        T=T, 
        num_samples=config["n_series"], 
        seq_len=max_len, 
        num_features=len(features)
    )

    num_samples = generated_scaled_samples.shape[0]
    generated_scaled_samples_flat = generated_scaled_samples.view(-1, len(features)).numpy()
    descaled_samples_flat = scaler.inverse_transform(generated_scaled_samples_flat)
    descaled_samples = descaled_samples_flat.reshape(num_samples, max_len, len(features))

    all_synthetic_dfs = []
    for i in range(num_samples):
        df_single_series = pd.DataFrame(descaled_samples[i], columns=features)
        df_single_series['ID'] = f'synthetic_{i}' 
        df_single_series['Cycle_Index'] = range(1, len(df_single_series) + 1) 
        all_synthetic_dfs.append(df_single_series)
    
    df_synth = pd.concat(all_synthetic_dfs, ignore_index=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()