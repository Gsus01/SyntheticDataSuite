import argparse
import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def correct_rul_column(df, rul_column='RUL', id_column='ID'):
    if rul_column in df.columns:
        df[rul_column] = df.groupby(id_column)[rul_column].transform(lambda x: np.maximum.accumulate(x[::-1])[::-1])
        logging.info("Columna RUL corregida para decrecer con el ciclo.")
    return df



def generate_series(generator, n_series, series_length, start_id=100, rul_column='RUL'):
    generated_data = []
    
    # Obtener los bin_edges para RUL del generador (modelo entrenado)
    rul_bin_edges = generator.bin_edges.get(rul_column)
    
    if rul_bin_edges is None:
        logging.error(f"No se encontraron bin_edges para la columna '{rul_column}'. No se puede re-cuantificar para corrección numérica.")
        raise ValueError(f"bin_edges para '{rul_column}' no encontrados. Asegúrate de que la columna se discretizó durante el entrenamiento y que el nombre es correcto.")

    num_bins = len(rul_bin_edges) - 1
    # Asegúrate de que las labels coinciden con cómo se generaron en el entrenamiento.
    # Si 'discretization_quantiles' fue un entero, las etiquetas serán 'Q1', 'Q2', etc.
    # Si fue 'labels=False' en pd.qcut, entonces las etiquetas son rangos como '(X, Y]'.
    # Para la simulación, 'pgmpy' devolverá las etiquetas que almacenó en el modelo.
    # Lo más seguro es que tus etiquetas sean de la forma 'Qx'
    labels = [f'Q{i+1}' for i in range(num_bins)] # Esta es la suposición más común

    rul_category_to_value = {}
    for i, label in enumerate(labels):
        if i < num_bins:
            lower_bound = rul_bin_edges[i]
            upper_bound = rul_bin_edges[i+1]
            
            if np.isinf(lower_bound):
                mid_point = upper_bound - (upper_bound - rul_bin_edges[i+2]) / 2 if i+2 < len(rul_bin_edges) else upper_bound / 2
                mid_point = min(mid_point, upper_bound) # Asegurar que no exceda el límite superior
            elif np.isinf(upper_bound):
                mid_point = lower_bound + (rul_bin_edges[i-1] - lower_bound) / 2 if i-1 >= 0 else lower_bound * 1.1
                mid_point = max(mid_point, lower_bound) # Asegurar que no sea menor que el límite inferior
            else:
                mid_point = (lower_bound + upper_bound) / 2
            
            rul_category_to_value[label] = mid_point
        else: # Caso borde para el último bin si hay desajuste
            rul_category_to_value[label] = rul_bin_edges[-1] 

    for i in range(n_series):
        sampled = generator.model.simulate(series_length)
        
        # 1. Convertir la columna RUL de categórica a numérica usando el mapeo
        if rul_column in sampled.columns and pd.api.types.is_categorical_dtype(sampled[rul_column]):
            # Aplicar el mapeo
            sampled[rul_column] = sampled[rul_column].map(rul_category_to_value)
            
            # AHORA, CONVERTIR EXPLÍCITAMENTE A TIPO NUMÉRICO (float)
            try:
                sampled[rul_column] = pd.to_numeric(sampled[rul_column], errors='coerce')
            except Exception as e:
                logging.error(f"Error al convertir la columna '{rul_column}' a numérica en la serie {start_id + i}: {e}")
                raise

            # Manejar posibles NaN después del mapeo y conversión
            if sampled[rul_column].isnull().any():
                logging.warning(f"Se generaron categorías de RUL no encontradas o mapeadas a NaN para la serie {start_id + i}. Rellenando NaNs con la media de la serie.")
                sampled[rul_column] = sampled[rul_column].fillna(sampled[rul_column].mean())
                # Si después de llenar sigue habiendo NaNs (ej. si toda la columna es NaN), usar 0 o un valor por defecto
                if sampled[rul_column].isnull().any():
                     sampled[rul_column] = sampled[rul_column].fillna(0.0) # Fallback a 0
            
        elif rul_column not in sampled.columns:
            logging.warning(f"La columna '{rul_column}' no se encuentra en los datos simulados para la serie {start_id + i}. Se omitirá la corrección de RUL.")
        # No se necesita else aquí, si no es categórica, ya es numérica y no necesita mapeo.

        sampled[generator.id_col] = start_id + i
        sampled[generator.index_col] = range(1, series_length + 1)
        generated_data.append(sampled)

    df_final = pd.concat(generated_data, ignore_index=True)
    
    # Aplicar la corrección de RUL ahora que es numérica
    df_final = correct_rul_column(df_final, rul_column=rul_column, id_column=generator.id_col)
    
    return df_final


    generated_data = []
    
    
    # Obtener los bin_edges para RUL del generador (modelo entrenado)
    rul_bin_edges = generator.bin_edges.get(rul_column)
    
    if rul_bin_edges is None:
        # Esto ocurrirá si 'RUL' no fue discretizada o si no se encontraron sus bin_edges.
        # Si 'RUL' nunca fue discretizada (ej. se trató como string por unique_values_threshold),
        # entonces ya sería string o numérica, y esta parte no aplicaría.
        # En este caso, el error original de TypeError significaría que fue discretizada
        # pero los bin_edges no se guardaron correctamente o el nombre de la columna no coincide.
        logging.error(f"No se encontraron bin_edges para la columna '{rul_column}'. No se puede re-cuantificar para corrección numérica.")
        raise ValueError(f"bin_edges para '{rul_column}' no encontrados. Asegúrate de que la columna se discretizó durante el entrenamiento y que el nombre es correcto.")

    # Crear un mapeo de categorías a valores numéricos (usando el punto medio del bin)
    # Asumimos que las etiquetas son 'Q1', 'Q2', ..., 'Qn' si 'discretization_quantiles' es un entero.
    # Si tus etiquetas fueran rangos como '(0.0, 10.0]', necesitarías una lógica de parsing más robusta.
    num_bins = len(rul_bin_edges) - 1
    labels = [f'Q{i+1}' for i in range(num_bins)] # Genera Q1, Q2, Q3...
    
    rul_category_to_value = {}
    for i, label in enumerate(labels):
        if i < num_bins: # Asegura que no nos salimos del índice de bin_edges
            lower_bound = rul_bin_edges[i]
            upper_bound = rul_bin_edges[i+1]
            
            # Simple heurística para los límites infinitos del primer/último bin
            if np.isinf(lower_bound):
                mid_point = upper_bound * 0.9 if upper_bound > 0 else upper_bound + 1 # Estimar un valor bajo
            elif np.isinf(upper_bound):
                mid_point = lower_bound * 1.1 if lower_bound > 0 else lower_bound - 1 # Estimar un valor alto
            else:
                mid_point = (lower_bound + upper_bound) / 2
            
            rul_category_to_value[label] = mid_point
        else: # Último bin si los labels exceden el número de límites (caso borde)
            rul_category_to_value[label] = rul_bin_edges[-1] # Usar el último límite conocido

    # --- FIN DE LA PREPARACIÓN DEL MAPEADOR ---

    for i in range(n_series):
        sampled = generator.model.simulate(series_length)
        
        # --- APLICACIÓN DEL CAMBIO DENTRO DEL BUCLE DE GENERACIÓN ---
        # 1. Convertir la columna RUL de categórica a numérica usando el mapeo
        if rul_column in sampled.columns and pd.api.types.is_categorical_dtype(sampled[rul_column]):
            sampled[rul_column] = sampled[rul_column].map(rul_category_to_value)
            
            # Manejar posibles NaN si alguna categoría generada no se mapea (poco común pero posible)
            if sampled[rul_column].isnull().any():
                logging.warning(f"Se generaron categorías de RUL no encontradas en el mapeo para la serie {start_id + i}. Rellenando NaNs con la media de la serie.")
                # Rellena NaN con la media de los valores numéricos ya mapeados en la serie actual
                sampled[rul_column] = sampled[rul_column].fillna(sampled[rul_column].mean())
        elif rul_column not in sampled.columns:
            logging.warning(f"La columna '{rul_column}' no se encuentra en los datos simulados para la serie {start_id + i}.")
        # --- FIN DE LA APLICACIÓN DEL CAMBIO ---

        sampled[generator.id_col] = start_id + i
        sampled[generator.index_col] = range(1, series_length + 1)
        generated_data.append(sampled)

    df_final = pd.concat(generated_data, ignore_index=True)
    
    # Aplicar la corrección de RUL ahora que es numérica (esta línea se mantiene)
    df_final = correct_rul_column(df_final, rul_column=rul_column, id_column=generator.id_col)
    
    return df_final

def main():
    parser = argparse.ArgumentParser(description="Genera series sintéticas a partir de un modelo bayesiano temporal.")
    parser.add_argument("--model_path", required=True, help="Ruta del archivo de entrada (.csv o .txt)")
    parser.add_argument("--output_data_path", required=True, help="Ruta donde guardar el modelo")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración JSON.")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logging.error(f"El archivo de configuración no se encontró en: {args.config}")
        exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    required_keys = [ "n_series", "series_length", "start_id"]
    for key in required_keys:
        if key not in config:
            logging.error(f"Falta la clave requerida '{key}' en el archivo de configuración JSON.")
            exit(1)

    model_path = args.model_path
    output_data_path = args.output_data_path
    n_series = config["n_series"]
    series_length = config["series_length"]
    start_id = config["start_id"]

    if not Path(model_path).exists():
        logging.error(f"El archivo del modelo no se encontró en: {model_path}")
        exit(1)

    logging.info("Cargando modelo bayesiano temporal...")
    generator = joblib.load(model_path)

    logging.info(f"Generando {n_series} series sintéticas de longitud {series_length}...")
    df_generated = generate_series(generator, n_series, series_length, start_id)

    output_dir = Path(output_data_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df_generated.to_csv(output_data_path, index=False)
    logging.info(f"Datos sintéticos guardados exitosamente en {output_data_path}")

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

if __name__ == "__main__":
    main()
