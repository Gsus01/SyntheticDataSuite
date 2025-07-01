
import argparse
import json
import pandas as pd
import joblib
import logging
from pathlib import Path
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class GeneticTimeSeriesSynthesizer:
    def __init__(self, min_length=1000, max_length=1100, population_size=50, generations=30, mutation_rate=0.1):
        self.min_length = min_length
        self.max_length = max_length
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Estos atributos se llenarán durante el método .fit()
        self.label_encoders = {}
        self.constant_columns = {}
        self.columns = None # Para mantener el orden original de las columnas
        self.time_col = None
        self.id_col = None
        self.feature_cols = None # Las columnas de características que realmente se usaron
        self.real_sequences = None # ¡Las secuencias de datos reales preprocesadas!

        self.id_start = 100 # Punto de inicio para los IDs de las series sintéticas

    def _preprocess(self, df, time_col, id_col):
        df_copy = df.copy() # Trabajar con una copia para no modificar el DF original
        self.columns = df_copy.columns.tolist() # Guarda el orden original de las columnas
        self.time_col = time_col
        self.id_col = id_col

        feature_cols = [col for col in df_copy.columns if col not in [time_col, id_col]]

        # Resetea y guarda columnas constantes
        self.constant_columns = {}
        for col in feature_cols:
            if df_copy[col].nunique() == 1:
                self.constant_columns[col] = df_copy[col].iloc[0]
        feature_cols = [col for col in feature_cols if col not in self.constant_columns]

        # Resetea y guarda LabelEncoders para columnas categóricas
        self.label_encoders = {}
        for col in df_copy[feature_cols].select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            self.label_encoders[col] = le

        self.feature_cols = feature_cols # Guarda las feature_cols finales
        return df_copy, feature_cols

    def _postprocess(self, df_synthetic_features):
        # Asegúrate de que el DataFrame de entrada ya tenga las columnas feature_cols
        df = df_synthetic_features.copy()

        # Añadir columnas constantes
        for col, val in self.constant_columns.items():
            df[col] = val

        # Decodificar categóricas
        for col, le in self.label_encoders.items():
            if col in df.columns: # Solo decodifica si la columna está presente
                # Redondea y convierte a int para usar con inverse_transform
                # Usa .clip para asegurar que los valores estén dentro del rango de clases
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
        """Genera una sola secuencia a partir de la población."""
        parent1 = random.choice(population_pool)
        parent2 = random.choice(population_pool)
        child = self._crossover(parent1, parent2)
        child = self._mutate(child)
        return child

    def fit(self, df_real, time_col='Cycle_Index', id_col='ID'):
        """
        Entrena el generador con los datos reales.
        Guarda la información de preprocesamiento necesaria para la generación.
        """
        logging.info("Iniciando fase de entrenamiento (fit) del modelo.")
        df_clean, feature_cols_processed = self._preprocess(df_real, time_col, id_col)
        self.feature_cols = feature_cols_processed # Guarda las feature_cols

        grouped = df_clean.groupby(id_col)
        self.real_sequences = [group[self.feature_cols].values for _, group in grouped if len(group) >= self.min_length]

        if not self.real_sequences:
            raise ValueError(f"No se encontraron secuencias reales con longitud mínima de {self.min_length} en los datos de entrada.")
        
        logging.info(f"Modelo entrenado exitosamente. {len(self.real_sequences)} secuencias reales válidas almacenadas.")


    def generate(self, n_sequences=5):
        """
        Genera series temporales sintéticas. Requiere que el modelo haya sido entrenado previamente
        para tener acceso a 'real_sequences' y la información de preprocesamiento.
        """
        if self.real_sequences is None or not self.real_sequences:
            raise ValueError("El modelo no ha sido entrenado o no contiene secuencias reales válidas. Llama a .fit() primero.")
        if not self.feature_cols or self.time_col is None or self.id_col is None:
            raise ValueError("Información de columnas o configuración de preprocesamiento faltante. Asegúrate de que el modelo fue entrenado correctamente.")

        synthetic_data = []

        logging.info(f"Generando {n_sequences} series sintéticas.")
        for i in range(n_sequences):
            new_id = self.id_start + i
            length = random.randint(self.min_length, self.max_length)

            # Inicializar población a partir de las secuencias reales almacenadas
            # Asegurarse de que random.choice(self.real_sequences) devuelva un array no vacío
            # y que np.random.choice pueda elegir elementos de él.
            # Se ha mejorado la selección para asegurar que siempre haya una secuencia de donde elegir
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

            # Si la secuencia final está vacía (posible con longitudes muy pequeñas o errores)
            if final_seq.size == 0:
                logging.warning(f"Secuencia generada para ID {new_id} está vacía. Saltando.")
                continue

            df_seq = pd.DataFrame(final_seq, columns=self.feature_cols)
            df_seq[self.time_col] = np.arange(1, len(df_seq) + 1)
            df_seq[self.id_col] = new_id

            df_seq = self._postprocess(df_seq)
            synthetic_data.append(df_seq)
        
        if not synthetic_data:
            raise ValueError("No se pudieron generar series sintéticas válidas.")

        df_synthetic = pd.concat(synthetic_data, ignore_index=True)
        # Reordenar columnas a la estructura original
        if self.columns:
            return df_synthetic[self.columns]
        else:
            return df_synthetic # Fallback si self.columns no se guardó por alguna razón


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo genético para series temporales y lo guarda.")
    parser.add_argument("--input_path", required=True, help="Ruta al archivo CSV de entrada con datos reales.")
    parser.add_argument("--output_path", required=True, help="Ruta para guardar el modelo entrenado (.joblib).")
    parser.add_argument("--config", required=True, help="Ruta al archivo JSON con configuración del modelo.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"No se encuentra el archivo de configuración: {config_path}")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    required_config_params = [
        "min_length", "max_length", "population_size",
        "generations", "mutation_rate", "time_col", "id_col"
    ]
    for r in required_config_params:
        if r not in config:
            logging.error(f"Falta el parámetro requerido en el JSON de configuración: {r}")
            exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        logging.error(f"No se encuentra el archivo de entrada de datos reales: {input_path}")
        exit(1)

    df_real = pd.read_csv(input_path)
    logging.info(f"Datos reales cargados desde {input_path}. Filas: {len(df_real)}")

    generator = GeneticTimeSeriesSynthesizer(
        min_length=config["min_length"],
        max_length=config["max_length"],
        population_size=config["population_size"],
        generations=config["generations"],
        mutation_rate=config["mutation_rate"]
    )

    try:
        # Aquí se entrena el modelo con los datos reales y se guarda la información necesaria en el objeto 'generator'
        generator.fit(df_real, time_col=config["time_col"], id_col=config["id_col"])
    except ValueError as e:
        logging.error(f"Error durante el entrenamiento del modelo: {e}")
        exit(1)
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Directorio creado: {output_dir}")

    # Guarda el modelo entrenado usando el protocolo más alto para evitar errores de carga
    joblib.dump({"generator": generator}, args.output_path, protocol=-1)
    logging.info(f"Modelo genético entrenado y guardado en {args.output_path}")

if __name__ == "__main__":
    main()