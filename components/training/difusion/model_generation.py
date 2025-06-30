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

# Configurar logging (puede ir aquí o dentro de main)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Modelo
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, max_len, model_dim=64, num_heads=4, num_layers=3): # <--- Add max_len here
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        # Use the passed max_len
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

    # Move the training logic into the class or a dedicated function
    # Adding a simple train method to the model class as suggested
    def train_model(self, dataloader, T, lr, num_epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        noise_schedule = torch.linspace(1e-4, 0.02, T)
        
        logging.info(f"Iniciando entrenamiento por {num_epochs} épocas con LR={lr}...")

        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch = batch[0] # Unpack the tensor from the dataset
                
                # Ensure batch is on the correct device if using GPU
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # batch = batch.to(device) 
                
                current_batch_size = batch.shape[0]
                t = torch.randint(0, T, (current_batch_size,)).to(batch.device) # Move t to device
                
                # Corrected diffusion process: return original noise for loss
                noisy_batch, target_noise = self.diffusion_process(batch, t, noise_schedule) 
                
                predicted_noise = self(noisy_batch) # Use self(noisy_batch) for forward pass
                
                loss = F.mse_loss(predicted_noise, target_noise) # Compare predicted noise with actual noise
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Época {epoch+1}/{num_epochs}, Pérdida: {avg_loss:.4f}")

    # Process of diffusion (moved into the class as a helper method)
    def diffusion_process(self, x, t, noise_schedule):
        alpha = 1 - noise_schedule[t]
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        noise = torch.randn_like(x)
        return torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise, noise # Return noise as well

# Define preprocess_data function
def preprocess_data(df, features_list, max_len):
    time_series_list = []
    for id_val in df['ID'].unique():
        # Check for existence of features_list columns
        missing_features = [f for f in features_list if f not in df.columns]
        if missing_features:
            logging.error(f"Características faltantes en el DataFrame: {missing_features}")
            raise ValueError(f"Las siguientes características no se encontraron en el CSV: {missing_features}")

        ts = df[df['ID'] == id_val][features_list].values 
        # Check for NaN values in the selected time series
        if np.isnan(ts).any():
            logging.warning(f"Se encontraron valores NaN para ID: {id_val}. Considere la imputación.")
            # Simple imputation example (replace with your chosen method)
            # ts = np.nan_to_num(ts, nan=0.0) # Replace NaN with 0, or use a more sophisticated method
            raise ValueError(f"Datos nulos encontrados para ID: {id_val}. El modelo actual no los maneja.")
            
        time_series_list.append(ts)

    if not time_series_list:
        logging.error("No se encontraron series de tiempo válidas después de la filtración por ID.")
        raise ValueError("No hay datos para procesar.")

    scaler = StandardScaler()
    all_data = np.vstack(time_series_list)
    scaler.fit(all_data)

    scaled_data = [scaler.transform(ts) for ts in time_series_list]

    padded_data = []
    for ts in scaled_data:
        if ts.shape[1] != len(features_list):
            logging.error(f"Inconsistencia de dimensiones de características. Esperado {len(features_list)}, obtenido {ts.shape[1]}")
            raise ValueError("Número de características incorrecto después del escalado.")

        if len(ts) > max_len:
            logging.warning(f"Una serie de tiempo (longitud {len(ts)}) es más larga que max_len ({max_len}). Será truncada.")
            ts = ts[:max_len, :] # Truncate if longer

        padded_ts = np.pad(ts, ((0, max_len - len(ts)), (0, 0)), mode='constant')
        padded_data.append(padded_ts)
    
    padded_data = np.array(padded_data)
    if padded_data.size == 0:
        logging.error("El array de datos padded está vacío. Revisar el preprocesamiento.")
        raise ValueError("Array de datos preprocesado vacío.")

    return torch.tensor(padded_data, dtype=torch.float32), scaler


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de difusión para series temporales.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output_path", required=True, help="Ruta para guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    required = [
        "features", "max_len",
        "T", "batch_size", "lr", "epochs"
    ]
    for key in required:
        if key not in config:
            logging.error(f"Falta la clave requerida en el JSON: {key}")
            exit(1)

    # Extract config parameters
    features = config["features"]
    max_len = config["max_len"]
    T = config["T"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"Archivo de entrada no encontrado: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    logging.info(f"Datos cargados desde {input_path}, forma: {df.shape}")

    logging.info("Preprocesando datos...")
    # Pass features from config to preprocess_data
    sequences_tensor, scaler = preprocess_data(df, features, max_len)

    # Dataset and DataLoader (moved into main)
    dataset = TensorDataset(sequences_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logging.info("Entrenando modelo de difusión...")
    model = DiffusionModel(
        input_dim=len(features), # Use len(features) for input_dim
        max_len=max_len,         # Pass max_len to the model's __init__
        model_dim=64,            # Default or from config
        num_heads=4,             # Default or from config
        num_layers=3             # Default or from config
    )
    
    # Train the model using the method defined in the class
    model.train_model(
        dataloader,              # Pass the DataLoader
        T=T,                     # Pass T
        lr=lr,                   # Pass learning rate
        num_epochs=epochs        # Pass number of epochs
    )

    # Ensure output directory exists:
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Directorio creado: {output_dir}")
    
    # Save model state_dict and scaler
    torch.save({
        'model_state_dict': model.state_dict(), # <--- Use model.state_dict()
        'scaler': scaler,
        'features': features,
        'max_len': max_len,
        'T': T
    }, args.output_path) # Use torch.save for the dictionary

    logging.info(f"Modelo y scaler guardados en {args.output_path}")


if __name__ == "__main__":
    main()