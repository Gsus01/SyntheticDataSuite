import pandas as pd
import json
import argparse
import os
import joblib
from pathlib import Path
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TimeSeriesBayesianGenerator:
    def __init__(self, df, id_col='ID', index_col='Cycle_Index', discretization_quantiles=3, unique_values_threshold=5):
        self.df = df.copy()
        # Nombres de columnas configurables
        self.id_col = id_col
        self.index_col = index_col
        self.model = None
        self.original_columns = [col for col in df.columns if col not in [id_col, index_col]]
        self.bin_edges = {}
        self.binary_vars = []
        self.one_hot_groups = {}
        self.discretized_df = None
        # Hiperparámetros de preprocesamiento configurables
        self.discretization_quantiles = discretization_quantiles
        self.unique_values_threshold = unique_values_threshold

    def _preprocess(self, df):
        df = df.copy()
        
        # Identificar columnas booleanas que podrían ser parte de un one-hot encoding
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        # Identificar columnas numéricas. Usamos pd.api.types.is_numeric_dtype para mayor robustez.
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Potenciales one-hot: numéricas que solo contienen 0 y 1, y tienen 2 valores únicos
        potential_onehot = [
            col for col in numeric_cols
            if set(df[col].dropna().unique()).issubset({0, 1}) and df[col].nunique() == 2
        ]

        self.one_hot_groups = {}
        for col in potential_onehot + bool_cols:
            # Intentar agrupar columnas con un prefijo común (ej. 'Status_Active', 'Status_Inactive')
            # Si el prefijo es 'Status', se agruparán bajo 'Status'
            prefix = col.split('_')[0] if '_' in col else col # Usar la columna completa si no hay '_'
            self.one_hot_groups.setdefault(prefix, []).append(col)

        for prefix, cols in self.one_hot_groups.items():
            # Si hay más de una columna en el grupo y su suma por fila es 1, se asume que es un one-hot encoding
            # y se convierte a una única columna categórica.
            if len(cols) > 1 and df[cols].sum(axis=1).eq(1).all():
                # Crea una nueva columna con el nombre del prefijo, tomando el sufijo de la columna que es 1
                # Por ejemplo, si 'Status_Active' es 1, la nueva columna 'Status' tendrá el valor 'Active'
                df[prefix] = df[cols].idxmax(axis=1).str.split('_').str[-1]
                df.drop(columns=cols, inplace=True) # Elimina las columnas originales del one-hot
            elif len(cols) == 1 and set(df[cols[0]].dropna().unique()).issubset({0, 1}):
                # Si es una sola columna con 0 y 1, es una binaria simple
                self.binary_vars.append(cols[0])
            elif len(cols) > 1: # Si son múltiples columnas pero no forman un one-hot perfecto (suman != 1)
                logging.warning(f"Las columnas agrupadas {cols} bajo el prefijo '{prefix}' no parecen ser un one-hot encoding perfecto (la suma por fila no es 1). Se mantendrán separadas.")
                # Si no es un one-hot perfecto, no las agrupamos y las dejamos como estaban inicialmente
                if prefix in df.columns: # Si la columna 'prefix' se creó, la eliminamos para no duplicar o mezclar
                    df.drop(columns=[prefix], inplace=True)


        # Recopilar variables binarias que no fueron parte de un one-hot o ya se manejaron
        self.binary_vars.extend([
            col for col in df.columns
            if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1}) and col not in self.binary_vars
        ])
        
        # Identificar variables continuas: todas las que no son id, index, one-hot transformadas, binarias o categóricas (object)
        continuous_vars = list(
            set(df.columns) - set(self.one_hot_groups.keys()) - # Excluir las nuevas columnas 'prefix' de one-hot
            set(self.binary_vars) - set(df.select_dtypes(include=['object']).columns) # Excluir binarias y object
        )

        for var in continuous_vars:
            # Si una variable continua tiene pocos valores únicos ( configurable: unique_values_threshold), la trata como categórica.
            if df[var].nunique() < self.unique_values_threshold:
                df[var] = df[var].astype(str)
                logging.info(f"Variable '{var}' convertida a tipo string por tener menos de {self.unique_values_threshold} valores únicos.")
                continue
            try:
                # Discretizar variables continuas en 'discretization_quantiles' cuantiles (configurable)
                # Opciones para 'q':
                # - Un entero (ej. 3): divide en ese número de cuantiles.
                # - Una lista de valores [0, 0.5, 1]: para cuartiles, puedes poner [0, 0.25, 0.5, 0.75, 1].
                # Opciones para 'labels':
                # - False: utiliza los rangos de los bins como etiquetas.
                # - Lista de strings (ej. ['Low', 'Medium', 'High']): etiquetas personalizadas.
                discretized, bins = pd.qcut(
                    df[var],
                    q=self.discretization_quantiles,
                    labels=[f'Q{i+1}' for i in range(self.discretization_quantiles)] if isinstance(self.discretization_quantiles, int) else False, # Genera Q1, Q2, Q3... o usa rangos
                    retbins=True,
                    duplicates='drop' # Elimina bins duplicados si hay pocos valores únicos para el número de cuantiles
                )
                df[var] = discretized
                self.bin_edges[var] = bins
                logging.info(f"Variable '{var}' discretizada en {self.discretization_quantiles} cuantiles.")
            except ValueError as e:
                # Si pd.qcut falla (ej. todos los valores son iguales), la convierte a string.
                logging.warning(f"No se pudo discretizar la variable '{var}' (error: {e}). Se convertirá a tipo string.")
                df[var] = df[var].astype(str)

        # Asegurarse de que todas las columnas restantes de tipo 'object' sean de tipo 'str'
        # Esto es crucial para pgmpy que trabaja con datos categóricos como strings o pandas.Categorical
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
                logging.info(f"Variable '{col}' asegurada como tipo string.")

        return df

    def fit(self):
        logging.info(f"Iniciando el entrenamiento del modelo. Columnas ID: '{self.id_col}', Índice: '{self.index_col}'")
        data = self.df.drop(columns=[self.id_col, self.index_col])
        data = self._preprocess(data)
        self.discretized_df = data.copy()

        logging.info("Buscando la estructura del modelo bayesiano con HillClimbSearch y BIC...")
        scorer = BIC(data)
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method=scorer)
        logging.info(f"Estructura del modelo encontrada: {best_model.edges()}")

        all_vars = data.columns.tolist()
        missing = set(all_vars) - set(best_model.nodes())

        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(all_vars)
        self.model.add_edges_from(best_model.edges())
        # Añadir nodos que no fueron detectados por HillClimbSearch si los hubiera (generalmente no ocurre si all_vars está bien)
        for node in missing:
            self.model.add_node(node)
            logging.warning(f"Nodo '{node}' añadido manualmente al modelo ya que no fue detectado por HillClimbSearch.")

        logging.info("Estimando los parámetros del modelo (CPDs) con MaximumLikelihoodEstimator...")
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        logging.info("Modelo bayesiano entrenado exitosamente.")

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo bayesiano de series temporales.")
    parser.add_argument("--input_path", required=True, help="Ruta del archivo de entrada (.csv o .txt)")
    parser.add_argument("--output_path", required=True, help="Ruta donde guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    args = parser.parse_args()

    # Cargar la configuración desde el archivo JSON
    if not Path(args.config).exists():
        logging.error(f"El archivo de configuración no se encontró en: {args.config}")
        exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Validar que las claves necesarias están en la configuración
    required_keys = [
        "input_file_path", "output_model_path", "id_column_name",
        "index_column_name", "discretization_quantiles", "unique_values_threshold"
    ]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo de configuración JSON.")
            exit(1)

    input_path = args.input_path
    output_path = args.output_path
    id_col = config["id_column_name"]
    index_col = config["index_column_name"]
    discretization_quantiles = config["discretization_quantiles"]
    unique_values_threshold = config["unique_values_threshold"]

    # Verificar que el archivo de entrada existe
    if not Path(input_path).exists():
        logging.error(f"El archivo de datos de entrada no se encontró en: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    logging.info(f"Archivo de datos CSV '{input_path}' cargado exitosamente.")

    logging.info("Creando generador de modelo bayesiano temporal...")
    generator = TimeSeriesBayesianGenerator(
        df,
        id_col=id_col,
        index_col=index_col,
        discretization_quantiles=discretization_quantiles,
        unique_values_threshold=unique_values_threshold
    )

    logging.info("Entrenando modelo bayesiano temporal...")
    generator.fit()

    # Asegurarse de que el directorio de salida existe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(generator, output_path)
    logging.info(f"Modelo guardado exitosamente en {output_path}")

if __name__ == "__main__":
    main()