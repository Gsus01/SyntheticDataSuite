import pandas as pd
import json
import argparse
import joblib
import logging
from pathlib import Path
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    RandomPermutation,
    ReversePermutation,
    MaskedAffineAutoregressiveTransform,
    AffineCouplingTransform
)
from nflows.nn.nets import ResidualNet


class TimeSeriesFlowSynthesizer:
    def __init__(self, id_column='ID', flow_type='maf', hidden_features=64, num_layers=5, use_reverse=True):
        self.id_column = id_column
        self.flow_type = flow_type.lower()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.use_reverse = use_reverse

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.constant_columns = {}
        self.columns = None
        self.flow = None

    def _build_transform(self, input_dim):
        transforms = []
        for _ in range(self.num_layers):
            if self.flow_type == 'maf':
                transform = MaskedAffineAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=self.hidden_features
                )
            elif self.flow_type == 'realnvp':
                transform = AffineCouplingTransform(
                    mask=torch.arange(0, input_dim) % 2,
                    transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                        in_features, out_features, hidden_features=self.hidden_features, num_blocks=2
                    )
                )
            else:
                raise ValueError(f"Unsupported flow type: {self.flow_type}")
            transforms.append(transform)
            transforms.append(RandomPermutation(features=input_dim))

        if self.use_reverse:
            transforms.insert(0, ReversePermutation(features=input_dim))

        return CompositeTransform(transforms)

    def _preprocess(self, df):
        df = df.copy()
        self.columns = [col for col in df.columns if col != self.id_column]

        # Guardar columnas constantes
        for col in self.columns:
            if df[col].nunique() == 1:
                self.constant_columns[col] = df[col].iloc[0]
        df = df.drop(columns=self.constant_columns.keys())

        # Codificar categóricas
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Escalar numéricas
        df[self.columns] = self.scaler.fit_transform(df[self.columns])
        return df

    def _postprocess(self, df_generated, original_len):
        df_out = pd.DataFrame(self.scaler.inverse_transform(df_generated), columns=self.columns)

        # Decodificar categóricas
        for col, le in self.label_encoders.items():
            df_out[col] = np.round(df_out[col]).astype(int)
            df_out[col] = le.inverse_transform(df_out[col].clip(0, len(le.classes_) - 1))

        # Añadir columnas constantes
        for col, val in self.constant_columns.items():
            df_out[col] = val

        # Añadir ID sintético
        ids = []
        for i in range(original_len):
            ids.extend([100 + i] * (df_out.shape[0] // original_len))
        df_out[self.id_column] = ids

        # Reordenar
        final_columns = self.columns + list(self.constant_columns.keys())
        return df_out[[self.id_column] + final_columns]

    def fit(self, df, batch_size=128, epochs=100, lr=1e-3):
        df_clean = self._preprocess(df)
        df_features = df_clean.drop(columns=self.id_column)

        data_tensor = torch.tensor(df_features.values, dtype=torch.float32)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = df_features.shape[1]
        transform = self._build_transform(input_dim)
        distribution = StandardNormal([input_dim])
        self.flow = Flow(transform, distribution)

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)

        self.flow.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                loss = -self.flow.log_prob(x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def generate(self, n_series=10, length_per_series=180):
        total_samples = n_series * length_per_series
        self.flow.eval()
        with torch.no_grad():
            samples = self.flow.sample(total_samples).numpy()
        df_synth = self._postprocess(samples, original_len=n_series)
        return df_synth

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas usando un modelo de normalizing flows.")
    parser.add_argument("--model_path", required=True, help="Ruta al modelo .pkl entrenado con Sparse Coding")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar los datos sintéticos")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # El 'start_id' no se pasa al método generate, pero lo mantenemos en config
    required_keys = ["n_series", "length_per_series", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo de configuración.")
            exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"El archivo del modelo no existe: {model_path}")
        exit(1)

    model = joblib.load(model_path)
    logging.info("Modelo cargado correctamente.")

    # --- CAMBIO CLAVE AQUÍ: Llamar a 'generate' en lugar de 'sample' ---
    # Y pasar solo los argumentos que 'generate' espera
    df_synth = model.generate(
        n_series=config["n_series"],
        length_per_series=config["length_per_series"]
        # No se pasa start_id aquí, ya que generate no lo acepta como argumento.
        # El manejo de ID sintéticos se hace internamente en _postprocess.
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()
