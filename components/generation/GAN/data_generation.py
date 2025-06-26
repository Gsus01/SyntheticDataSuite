import pandas as pd
import argparse
import json
import joblib
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Generator(nn.Module):
    def __init__(self, noise_dim, seq_len, feature_dim=1, hidden_dim=64):
        """
        noise_dim: dimensionalidad del vector aleatorio de entrada
        seq_len: longitud de la serie temporal generada
        feature_dim: número de variables/features por paso de tiempo
        hidden_dim: tamaño de la capa oculta
        """
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, seq_len * feature_dim),
            nn.Tanh()  # asume datos escalados en [-1,1]
        )

    def forward(self, z):
        # z: [batch_size, noise_dim]
        x = self.net(z)
        # reshape a [batch_size, seq_len, feature_dim]
        return x.view(-1, self.seq_len, self.feature_dim)

class Discriminator(nn.Module):
    def __init__(self, seq_len, feature_dim=1, hidden_dim=64):
        """
        seq_len: longitud de la serie temporal de entrada
        feature_dim: número de variables/features por paso de tiempo
        hidden_dim: tamaño de la capa oculta
        """
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feature_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        return self.net(x).view(-1)

class TimeSeriesGAN:
    def __init__(self, seq_len, feature_dim=1, noise_dim=100, hidden_dim=64, lr=2e-4, device=None):
        """
        seq_len: longitud de la serie temporal
        feature_dim: número de variables
        noise_dim: dimensionalidad del ruido de entrada
        hidden_dim: tamaño común para hidden layers
        lr: learning rate para ambos optimizadores
        device: 'cpu' o 'cuda'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = Generator(noise_dim, seq_len, feature_dim, hidden_dim).to(self.device)
        self.D = Discriminator(seq_len, feature_dim, hidden_dim).to(self.device)
        self.noise_dim = noise_dim

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, data_loader, epochs=100, print_every=10):
        self.G.train()
        self.D.train()
        for epoch in range(1, epochs+1):
            loss_D_epoch = 0.0
            loss_G_epoch = 0.0
            for (real_seq,) in data_loader:      # <— desempaquetamos el tensor
                real_seq = real_seq.to(self.device).float()
                batch_size = real_seq.size(0)
                valid = torch.ones(batch_size, device=self.device)
                fake_label = torch.zeros(batch_size, device=self.device)

                # ——— entrena Discriminador ———
                self.optim_D.zero_grad()
                pred_real = self.D(real_seq)
                loss_real = self.criterion(pred_real, valid)
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_seq = self.G(z)
                pred_fake = self.D(fake_seq.detach())
                loss_fake = self.criterion(pred_fake, fake_label)
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optim_D.step()

                # ——— entrena Generador ———
                self.optim_G.zero_grad()
                pred_fake2 = self.D(fake_seq)
                loss_G = self.criterion(pred_fake2, valid)
                loss_G.backward()
                self.optim_G.step()

                loss_D_epoch += loss_D.item()
                loss_G_epoch += loss_G.item()

            if epoch % print_every == 0 or epoch == 1:
                avg_D = loss_D_epoch / len(data_loader)
                avg_G = loss_G_epoch / len(data_loader)
                print(f"Epoch {epoch:3d}/{epochs} | D Loss: {avg_D:.4f} | G Loss: {avg_G:.4f}")


    def generate(self, num_samples):
        """Genera secuencias sintéticas."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim, device=self.device)
            synthetic = self.G(z)
        return synthetic.cpu().numpy()
    
def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas con un modelo GAN entrenado.")
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pkl entrenado con Sparse Coding")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Añadimos "seq_len" y "feature_dim" a las claves requeridas para la validación
    required = ["n_series", "series_length", "start_id", "target_columns", "seq_len", "feature_dim"] 
    for r in required:
        if r not in config:
            logging.error(f"Falta la clave requerida en config: {r}")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"Modelo no encontrado en: {model_path}")
        exit(1)
    
    # --- LEER seq_len y feature_dim DIRECTAMENTE DEL CONFIG ---
    model_seq_len = config["seq_len"]
    model_feature_dim = config["feature_dim"]
    # --------------------------------------------------------

    # Cargamos el modelo GAN entrenado
    gan_model = joblib.load(model_path)
    logging.info("Modelo GAN cargado correctamente.")

    logging.info(f"Generando {config['n_series']} series sintéticas de longitud {config['series_length']}...")
    
    # Llamamos a 'generate' con el número de series que queremos.
    # El modelo generará datos con la seq_len y feature_dim con las que fue entrenado.
    synthetic_data_array = gan_model.generate(num_samples=config["n_series"])

    # --- ELIMINA LAS SIGUIENTES DOS LÍNEAS, SON LA CAUSA DEL ERROR ---
    # model_seq_len = gan_model.seq_len 
    # model_feature_dim = gan_model.feature_dim
    # ---------------------------------------------------------------

    if config["series_length"] != model_seq_len:
        logging.warning(f"La 'series_length' en config ({config['series_length']}) no coincide con la longitud de secuencia del modelo (usando {model_seq_len}). Se generarán series con la longitud del modelo.")
        series_length_for_indexing = model_seq_len
    else:
        series_length_for_indexing = config["series_length"]

    reshaped_data = synthetic_data_array.reshape(-1, model_feature_dim)

    # Crear el DataFrame de Pandas
    df_synth = pd.DataFrame(reshaped_data, columns=config["target_columns"])

    # Añadir las columnas 'ID' y 'Cycle_Index'
    num_series = config["n_series"]
    start_id = config["start_id"]

    ids = np.repeat(np.arange(start_id, start_id + num_series), series_length_for_indexing)
    cycle_indices = np.tile(np.arange(1, series_length_for_indexing + 1), num_series)
    
    df_synth["ID"] = ids
    df_synth["Cycle_Index"] = cycle_indices

    output_cols = ["ID", "Cycle_Index"] + config["target_columns"]
    df_synth = df_synth[output_cols]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
