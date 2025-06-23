#!/usr/bin/env python3
"""
Microservicio de preprocesamiento de datos para series temporales
Ejecutable de forma aislada para workflows de Argo
"""

import argparse
import logging
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class TimeSeriesPreprocessor:
    """Procesador de series temporales configurable"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        

        
    def load_data(self, input_path: str) -> pd.DataFrame:
        """Carga los datos del archivo CSV"""
        try:
            df = pd.read_csv(input_path)
            self.logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            raise
            
    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detecta automáticamente la columna de fecha/tiempo"""
        datetime_col = self.config.get('data', {}).get('datetime_column')
        
        if datetime_col and datetime_col in df.columns:
            return datetime_col
            
        # Auto-detección
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                return col
                
        # Intentar detectar por tipo de datos
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                return col
            except:
                continue
                
        return None
        
    def preprocess_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa la columna de fecha/tiempo"""
        datetime_col = self.detect_datetime_column(df)
        datetime_format_string = self.config.get("data", {}).get("datetime_format")

        if not datetime_col:
            self.logger.warning("No se encontró columna de fecha/tiempo")
            return df

        try:
            if datetime_format_string:
                self.logger.info(f"Usando formato de fecha/tiempo especificado: {datetime_format_string} para columna {datetime_col}")
                df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format_string, errors="coerce")
            else:
                # Try to automatically parse, coercing errors to NaT.
                # pandas now uses strict datetime inference by default, so infer_datetime_format is no longer needed
                # by relying on errors='coerce' and then dropping NaT.
                self.logger.info(f"Intentando parseo automático de fecha/tiempo para columna {datetime_col}")
                df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

            # Drop rows where the primary datetime column could not be parsed (is NaT)
            original_rows = len(df)
            df.dropna(subset=[datetime_col], inplace=True)
            if len(df) < original_rows:
                self.logger.warning(f"{original_rows - len(df)} filas eliminadas debido a valores NaT en la columna de fecha/tiempo {datetime_col} después del intento de parseo.")

            if df.empty:
                self.logger.warning(f"DataFrame vacío después de procesar NaT en columna {datetime_col}. No se puede establecer índice.")
                return df # Return empty df, can't set index

            df = df.set_index(datetime_col).sort_index()
            self.logger.info(f"Columna temporal procesada: {datetime_col}")

        except Exception as e:
            self.logger.error(f"Error procesando columna temporal {datetime_col}: {e}")
            # Depending on desired robustness, might return df or raise.
            # Current logic returns df, potentially unprocessed for datetime.
            
        return df
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja los valores faltantes según configuración"""
        missing_config = self.config.get('preprocessing', {}).get('missing_values', {})
        strategy = missing_config.get('strategy', 'drop')
        
        if strategy == 'drop':
            # Eliminar filas con valores nulos
            threshold = missing_config.get('threshold', 0.5)
            df = df.dropna(thresh=int(len(df.columns) * threshold))
            self.logger.info(f"Filas eliminadas por valores nulos (umbral: {threshold})")
            
        elif strategy == 'fill':
            # Rellenar valores
            fill_method = missing_config.get('method', 'forward')
            if fill_method == 'forward':
                df = df.fillna(method='ffill')
            elif fill_method == 'backward':
                df = df.fillna(method='bfill')
            elif fill_method == 'interpolate':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].interpolate()
            elif fill_method == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                
            self.logger.info(f"Valores nulos rellenados con método: {fill_method}")
            
        return df
        
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona las características especificadas"""
        features_config = self.config.get('preprocessing', {}).get('features', {})
        
        # Si features_config es None, convertirlo a diccionario vacío
        if features_config is None:
            features_config = {}
        
        if 'include' in features_config and features_config['include'] is not None:
            include_cols = features_config['include']
            available_cols = [col for col in include_cols if col in df.columns]
            df = df[available_cols]
            self.logger.info(f"Características incluidas: {available_cols}")
            
        if 'exclude' in features_config and features_config['exclude'] is not None:
            exclude_cols = features_config['exclude']
            df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
            self.logger.info(f"Características excluidas: {exclude_cols}")
            
        return df
        
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y maneja outliers"""
        outliers_config = self.config.get('preprocessing', {}).get('outliers', {})
        
        if not outliers_config.get('enabled', False):
            return df
            
        method = outliers_config.get('method', 'iqr')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                threshold = outliers_config.get('threshold', 3)
                outliers_mask = z_scores > threshold
                
            # Manejo de outliers
            action = outliers_config.get('action', 'remove')
            if action == 'remove':
                df = df[~outliers_mask]
            elif action == 'cap':
                if method == 'iqr':
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    
        self.logger.info(f"Outliers procesados con método: {method}")
        return df
        
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza los datos numéricos"""
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        
        if not norm_config.get('enabled', False):
            return df
            
        method = norm_config.get('method', 'minmax')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        elif method == 'zscore':
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        elif method == 'robust':
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].median()) / (df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25))
            
        self.logger.info(f"Datos normalizados con método: {method}")
        return df
        
    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remuestrea los datos temporales"""
        resample_config = self.config.get('preprocessing', {}).get('resampling', {})
        
        if not resample_config.get('enabled', False):
            return df
            
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("No se puede remuestrear: índice no es temporal")
            return df
            
        frequency = resample_config.get('frequency', 'h') # Changed '1H' to 'h'
        method = resample_config.get('method', 'mean')
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns

        if method == 'mean':
            df_numeric = df[numeric_cols].resample(frequency).mean()
            df_non_numeric = df[non_numeric_cols].resample(frequency).first()
            df = pd.concat([df_numeric, df_non_numeric], axis=1)
        elif method == 'sum':
            df_numeric = df[numeric_cols].resample(frequency).sum()
            df_non_numeric = df[non_numeric_cols].resample(frequency).first()
            df = pd.concat([df_numeric, df_non_numeric], axis=1)
        elif method == 'first':
            df = df.resample(frequency).first()
        elif method == 'last':
            df = df.resample(frequency).last()
        # TODO: Consider how to handle cases where all columns are numeric or non-numeric to avoid empty DFs in concat
        # For now, this handles mixed types. If one group is empty, concat might behave unexpectedly or error.
        # A better approach for concat might be selective if len(numeric_cols) > 0 and len(non_numeric_cols) > 0.
            
        self.logger.info(f"Datos remuestreados a frecuencia: {frequency}, método: {method}")
        return df
        
    def create_sliding_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea ventanas deslizantes para análisis temporal"""
        windows_config = self.config.get('preprocessing', {}).get('sliding_windows', {})
        
        if not windows_config.get('enabled', False):
            return df
            
        window_size = windows_config.get('size', 10)
        features = windows_config.get('features', [])
        
        if not features:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            
        for feature in features:
            if feature in df.columns:
                # Media móvil
                df[f'{feature}_rolling_mean'] = df[feature].rolling(window=window_size).mean()
                # Desviación estándar móvil
                df[f'{feature}_rolling_std'] = df[feature].rolling(window=window_size).std()
                
        self.logger.info(f"Ventanas deslizantes creadas (tamaño: {window_size})")
        return df
        
    def process(self, input_path: str, output_path: str) -> None:
        """Ejecuta el pipeline completo de preprocesamiento"""
        self.logger.info("Iniciando preprocesamiento de datos")
        
        # Cargar datos
        df = self.load_data(input_path)
        original_shape = df.shape
        
        # Pipeline de preprocesamiento
        df = self.preprocess_datetime(df)
        df = self.select_features(df)
        df = self.handle_missing_values(df)
        df = self.detect_outliers(df)
        df = self.normalize_data(df)
        df = self.resample_data(df)
        df = self.create_sliding_windows(df)
        
        # Guardar resultados
        df.to_csv(output_path)
        final_shape = df.shape
        
        self.logger.info(f"Preprocesamiento completado")
        self.logger.info(f"Forma original: {original_shape} -> Forma final: {final_shape}")
        self.logger.info(f"Datos guardados en: {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga la configuración desde archivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error cargando configuración: {e}")
        sys.exit(1)


def main():

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    """Función principal"""
    parser = argparse.ArgumentParser(description='Preprocesador de series temporales')
    parser.add_argument('--config', required=True, help='Archivo de configuración YAML')
    parser.add_argument('--input', required=True, help='Archivo CSV de entrada')
    parser.add_argument('--output', required=True, help='Archivo CSV de salida')
    
    args = parser.parse_args()
    
    # Validar archivos
    if not Path(args.config).exists():
        logging.critical(f"Error: Archivo de configuración no encontrado: {args.config}")
        sys.exit(1)
        
    if not Path(args.input).exists():
        logging.critical(f"Error: Archivo de entrada no encontrado en el pod: {args.input}")
        sys.exit(1)
        
    # Crear directorio de salida si no existe
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Cargar configuración y procesar
    config = load_config(args.config)
    preprocessor = TimeSeriesPreprocessor(config)
    preprocessor.process(args.input, args.output)


if __name__ == "__main__":
    main()
