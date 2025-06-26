
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
    parser = argparse.ArgumentParser(description="Entrena un modelo GAN para series temporales.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada")
    parser.add_argument("--output_path", required=True, help="Ruta para guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Validar claves requeridas en la configuración
    required_config_keys = [
        "id_column", "time_column", "target_columns",
        "noise_dim", "epochs", "batch_size", "lr", # 'latent_dim' ya no se usa aquí
        "seq_len" 
    ]
    for r in required_config_keys:
        if r not in config:
            logging.error(f"Falta la clave requerida en config: {r}")
            exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"Archivo de entrada no encontrado: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    logging.info(f"Datos cargados: {df.shape}")

    # --- Preprocesamiento de Datos ANTES de pasar al GAN ---
    # 1. Manejo de Nulos (muy importante para GANs)
   
    original_rows = df.shape[0]
    df.dropna(subset=config["target_columns"], inplace=True)
    if df.shape[0] < original_rows:
        logging.warning(f"Se eliminaron {original_rows - df.shape[0]} filas con valores nulos.")

    # 2. Manejo de Categóricos y Escalado de Numéricos
    
    scaler = StandardScaler()
    label_encoders = {}
    
    numerical_cols = []
    categorical_cols = []
    
    # Identificar y preprocesar columnas
    for col in config["target_columns"]:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
            
    # Escalar columnas numéricas (GAN espera [-1, 1], StandardScaler produce ~N(0,1). Tanh puede ayudar, pero MinMaxScaler es más directo)
    
    if numerical_cols:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

   

    seq_len = config["seq_len"] # Debe estar en tu config
    feature_dim = len(config["target_columns"])

   
    grouped_data = [g[config["target_columns"]].values for _, g in df.groupby(config["id_column"])]
    
    
    processed_sequences = []
    for group_array in grouped_data:
        if len(group_array) >= seq_len:
            processed_sequences.append(group_array[:seq_len]) # Truncar
        else:
            # Puedes paddeas con ceros o un valor específico
            padding = np.zeros((seq_len - len(group_array), feature_dim))
            processed_sequences.append(np.vstack((group_array, padding))) # Paddear

    if not processed_sequences:
        logging.error("No se pudieron procesar secuencias válidas con la configuración dada.")
        exit(1)

    data_tensor = torch.tensor(np.array(processed_sequences), dtype=torch.float32)

    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    logging.info(f"Creado DataLoader con {len(data_loader)} batches de tamaño {config['batch_size']}.")
    
    # --- Inicialización del GAN ---
    gan = TimeSeriesGAN(
        seq_len=seq_len,
        feature_dim=feature_dim, # Número de columnas objetivo
        noise_dim=config["noise_dim"],
        hidden_dim=config.get("hidden_dim", 64), # Usar .get para valor por defecto si no está en config
        lr=config["lr"]
        # epochs, batch_size, id_column, time_column, target_columns YA NO VAN AQUI
    )

    logging.info("Entrenando modelo GAN...")
    
    gan.train(
        data_loader=data_loader,
        epochs=config["epochs"]
        
    )
    logging.info("Entrenamiento completado.")

    
    joblib.dump(gan, args.output_path)
    logging.info(f"Modelo GAN guardado en {args.output_path}")

if __name__ == "__main__":
    main()
