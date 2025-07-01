# generate_data.py
import argparse
import json
import pandas as pd
import joblib
import logging
from pathlib import Path
import random 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class GeneticTimeSeriesSynthesizer:
    def __init__(self, min_length=1000, max_length=1100, population_size=50, generations=30, mutation_rate=0.1):
        self.min_length = min_length
        self.max_length = max_length
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.label_encoders = {}
        self.constant_columns = {}
        self.columns = None
        self.time_col = None
        self.id_col = None
        self.feature_cols = None
        self.real_sequences = None

        self.id_start = 100

    def _preprocess(self, df, time_col, id_col):

        df_copy = df.copy() 
        self.columns = df_copy.columns.tolist() 
        self.time_col = time_col
        self.id_col = id_col

        feature_cols = [col for col in df_copy.columns if col not in [time_col, id_col]]

        self.constant_columns = {}
        for col in feature_cols:
            if df_copy[col].nunique() == 1:
                self.constant_columns[col] = df_copy[col].iloc[0]
        feature_cols = [col for col in feature_cols if col not in self.constant_columns]

        self.label_encoders = {}
        for col in df_copy[feature_cols].select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            self.label_encoders[col] = le

        self.feature_cols = feature_cols 
        return df_copy, feature_cols

    def _postprocess(self, df_synthetic_features):
        df = df_synthetic_features.copy()

        for col, val in self.constant_columns.items():
            df[col] = val

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = np.round(df[col]).astype(int)
                df[col] = le.inverse_transform(df[col].clip(0, len(le.classes_) - 1))
        return df

    def _crossover(self, parent1, parent2):
        cut = random.randint(1, len(parent1) - 1)
        return np.vstack((parent1[:cut], parent2[cut:]))

    def _mutate(self, seq):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(seq) - 1)
            dim = random.randint(0, seq.shape[1] - 1)
            noise = np.random.normal(0, 0.1)
            seq[idx, dim] += noise
        return seq

    def _generate_single_sequence_data(self, population_pool):
        parent1 = random.choice(population_pool)
        parent2 = random.choice(population_pool)
        child = self._crossover(parent1, parent2)
        child = self._mutate(child)
        return child

    def fit(self, df_real, time_col='Cycle_Index', id_col='ID'):
        logging.warning("El método .fit() fue llamado en el script de generación. No es necesario ni recomendado.")
        df_clean, feature_cols_processed = self._preprocess(df_real, time_col, id_col)
        self.feature_cols = feature_cols_processed 

        grouped = df_clean.groupby(id_col)
        self.real_sequences = [group[self.feature_cols].values for _, group in grouped if len(group) >= self.min_length]

        if not self.real_sequences:
            raise ValueError(f"No se encontraron secuencias reales con longitud mínima de {self.min_length} en los datos de entrada.")
        
    def generate(self, n_sequences=5):
        """
        Genera series temporales sintéticas. Requiere que el modelo haya sido entrenado previamente
        para tener acceso a 'real_sequences' y la información de preprocesamiento.
        """
        if self.real_sequences is None or not self.real_sequences:
            raise ValueError("El modelo no ha sido entrenado o no contiene secuencias reales válidas. Llama a .fit() primero en los datos originales.")
        if not self.feature_cols or self.time_col is None or self.id_col is None:
            raise ValueError("Información de columnas o configuración de preprocesamiento faltante. Asegúrate de que el modelo fue entrenado correctamente.")

        synthetic_data = []

        logging.info(f"Generando {n_sequences} series sintéticas.")
        for i in range(n_sequences):
            new_id = self.id_start + i
            length = random.randint(self.min_length, self.max_length)

            # Inicializar población a partir de las secuencias reales almacenadas
            real_seq_sample = random.choice(self.real_sequences)
            if real_seq_sample.size == 0:
                logging.warning(f"Secuencia real seleccionada está vacía. Saltando generación para ID {new_id}.")
                continue
            
            population = [real_seq_sample[np.random.choice(real_seq_sample.shape[0], length, replace=True)]
                          for _ in range(self.population_size)]

            # Evolución
            for gen in range(self.generations):
                new_population = [self._generate_single_sequence_data(population) for _ in range(self.population_size)]
                population = new_population
            
            # Selección final (el mejor por varianza)
            final_seq = max(population, key=lambda seq: np.var(seq) if seq.size > 0 else 0)

            if final_seq.size == 0:
                logging.warning(f"Secuencia generada para ID {new_id} está vacía. Saltando.")
                continue

            df_seq = pd.DataFrame(final_seq, columns=self.feature_cols)
            df_seq[self.time_col] = np.arange(1, len(df_seq) + 1)
            df_seq[self.id_col] = new_id

            df_seq = self._postprocess(df_seq)
            synthetic_data.append(df_seq)
        
        if not synthetic_data:
            raise ValueError("No se pudieron generar series sintéticas válidas. Revisa los parámetros del modelo.")

        df_synthetic = pd.concat(synthetic_data, ignore_index=True)
        if self.columns:
            return df_synthetic[self.columns]
        else:
            return df_synthetic


def main():
    parser = argparse.ArgumentParser(description="Carga un modelo genético entrenado y genera series sintéticas.")
    parser.add_argument("--model_path", required=True, help="Ruta al archivo .joblib del modelo entrenado.")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar las series sintéticas generadas (.csv).")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logging.error(f"Modelo no encontrado en: {model_path}")
        exit(1)

    try:
        loaded_data = joblib.load(model_path)
        generator = loaded_data["generator"]
    except Exception as e:
        logging.error(f"Error al cargar el modelo desde {model_path}: {e}")
        exit(1)

    if not isinstance(generator, GeneticTimeSeriesSynthesizer):
        logging.error("El archivo cargado no contiene un objeto GeneticTimeSeriesSynthesizer válido.")
        exit(1)
    
    logging.info(f"Modelo cargado desde {model_path}.")

    try:
        df_synthetic = generator.generate(n_sequences=config["n_series"])
        # Una comprobación más robusta para el id_col si existe
        num_generated_sequences = len(df_synthetic[generator.id_col].unique()) if hasattr(generator, 'id_col') and generator.id_col in df_synthetic.columns else args.n_sequences
        logging.info(f"Se generaron {num_generated_sequences} series sintéticas con un total de {len(df_synthetic)} filas.")
    except ValueError as e:
        logging.error(f"Error al generar series sintéticas: {e}")
        exit(1)
    except KeyError as e:
         logging.error(f"Error al acceder a la columna ID ('{generator.id_col}') durante la generación. Es posible que la información de columna no se haya guardado correctamente en el modelo o la columna no exista en los datos generados: {e}")
         exit(1)


    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)
    logging.info(f"Series sintéticas guardadas en {output_path}")

if __name__ == "__main__":
    main()