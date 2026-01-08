import pandas as pd
import numpy as np
import json
import argparse
import joblib
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder

DEFAULT_INPUT_DIR = "/data/inputs"
DEFAULT_OUTPUT_DIR = "/data/outputs"
DEFAULT_CONFIG_DIR = "/data/config"

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def discover_file(directory: Path, patterns, description: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado para {description}: {directory}")
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No se encontraron archivos {description} en {directory} (patrones: {', '.join(patterns)})")

class SeqVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=20):
        super(SeqVAE, self).__init__()
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, h = self.encoder_rnn(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h_dec = self.decoder_input(z).unsqueeze(0)
        h_dec_expanded = h_dec.repeat(seq_len, 1, 1).permute(1, 0, 2)
        out, _ = self.decoder_rnn(h_dec_expanded)
        return self.output_layer(out)

    def forward(self, x, seq_len):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, seq_len)
        return x_recon, mu, logvar
    
class TimeSeriesVAESynthesizer:
    def __init__(self, latent_dim=20, start_id=100, device=None):
        self.latent_dim = latent_dim
        self.start_id = start_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.label_encoders = {}
        self.model = None
        self.columns = []
        self.categorical_cols = []
        self.numerical_cols = []

    def _preprocess(self, df):
        df = df.copy()
        self.columns = [col for col in df.columns if col not in ['ID', 'Cycle_Index']]
        for col in self.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)
        self.scaler = StandardScaler()
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        return df

    def _postprocess(self, df):
        df[self.numerical_cols] = self.scaler.inverse_transform(df[self.numerical_cols])
        for col, le in self.label_encoders.items():
            df[col] = np.clip(df[col], 0, len(le.classes_) - 1).astype(int)
            df[col] = le.inverse_transform(df[col])
        return df

    def _pad_sequences(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len, len(self.columns)))
        for i, seq in enumerate(sequences):
            padded[i, :len(seq), :] = seq
        return padded, [len(seq) for seq in sequences]

    def fit(self, df, epochs=100, batch_size=16, lr=1e-3):
        df = self._preprocess(df)
        groups = [g[self.columns].values for _, g in df.groupby('ID')]
        padded_data, lengths = self._pad_sequences(groups)
        X = torch.tensor(padded_data, dtype=torch.float32).to(self.device)

        self.model = SeqVAE(input_dim=X.shape[2], latent_dim=self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        def loss_fn(recon, x, mu, logvar, lengths):
            mask = torch.zeros_like(x)
            for i, l in enumerate(lengths):
                mask[i, :l, :] = 1
            recon_loss = ((recon - x) ** 2 * mask).sum() / mask.sum()
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_div

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), batch_size):
                idx = perm[i:i+batch_size]
                batch = X[idx]
                lens = [lengths[j] for j in idx.tolist()]
                recon, mu, logvar = self.model(batch, max(lens))
                loss = loss_fn(recon, batch, mu, logvar, lens)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate(self, n_series=10, seq_len=100):
        self.model.eval()
        generated_data = []
        with torch.no_grad():
            for i in range(n_series):
                z = torch.randn(1, self.latent_dim).to(self.device)
                recon = self.model.decode(z, seq_len)[0].cpu().numpy()
                df_rec = pd.DataFrame(recon, columns=self.columns)
                df_rec = self._postprocess(df_rec)
                df_rec['Cycle_Index'] = np.arange(1, seq_len + 1)
                df_rec['ID'] = self.start_id + i
                cols = ['ID', 'Cycle_Index'] + self.columns
                generated_data.append(df_rec[cols])
        return pd.concat(generated_data, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas a partir de un modelo VAE.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directorio que contiene el modelo entrenado (.pkl)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio donde guardar los datos sintéticos")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR, help="Directorio con el archivo JSON de configuración")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
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

    # Las claves requeridas para la generación
    required = ["n_series", "series_length", "start_id"] # 'start_id' se usa en el constructor del sintetizador
    for r in required:
        if r not in config:
            logging.error(f"Falta la clave requerida en config: {r}")
            exit(1)

    vae_model = joblib.load(model_path)
    logging.info(f"Modelo cargado correctamente desde {model_path}")

    logging.info(f"Generando {config['n_series']} series de longitud {config['series_length']}...")
    

    df_synthetic = vae_model.generate(
        n_series=config["n_series"],
        seq_len=config["series_length"]
        
    )

    output_path = output_dir / f"synthetic_{model_path.stem}.csv"
    df_synthetic.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
