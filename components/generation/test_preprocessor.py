#!/usr/bin/env python3
"""
Tests unitarios para el preprocesador de series temporales
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import yaml
import os
from pathlib import Path
import sys

# Añadir el directorio padre al path para importar el módulo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import TimeSeriesPreprocessor


class TestTimeSeriesPreprocessor(unittest.TestCase):
    """Tests para el preprocesador de series temporales"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_config = {
            'logging': {'level': 'WARNING'},
            'data': {'datetime_column': None},
            'preprocessing': {
                'features': {},
                'missing_values': {
                    'strategy': 'fill',
                    'method': 'interpolate'
                },
                'outliers': {
                    'enabled': True,
                    'method': 'iqr',
                    'action': 'cap'
                },
                'normalization': {
                    'enabled': True,
                    'method': 'minmax'
                },
                'resampling': {
                    'enabled': False
                },
                'sliding_windows': {
                    'enabled': True,
                    'size': 3
                }
            }
        }
        self.preprocessor = TimeSeriesPreprocessor(self.test_config)
        
    def test_sensor_data_with_nulls(self):
        """Test con datos de sensores que tienen valores nulos"""
        input_file = 'test_data/sensor_data_with_nulls.csv'
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            output_file = tmp.name
            
        try:
            # Procesar datos
            self.preprocessor.process(input_file, output_file)
            
            # Verificar que se generó el archivo
            self.assertTrue(os.path.exists(output_file))
            
            # Cargar y verificar resultados
            result_df = pd.read_csv(output_file)
            
            # Verificar que no hay valores nulos en columnas numéricas
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            null_counts = result_df[numeric_cols].isnull().sum()
            
            # Debería haber muy pocos o ningún valor nulo después del procesamiento
            total_nulls = null_counts.sum()
            self.assertLessEqual(total_nulls, len(result_df) * 0.1, 
                               "Demasiados valores nulos después del procesamiento")
            
            print(f"✓ Sensor data: {len(result_df)} filas procesadas, {total_nulls} valores nulos restantes")
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_financial_data_with_outliers(self):
        """Test con datos financieros que tienen outliers"""
        input_file = 'test_data/financial_data_with_outliers.csv'
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            output_file = tmp.name
            
        try:
            # Procesar datos
            self.preprocessor.process(input_file, output_file)
            
            # Verificar que se generó el archivo
            self.assertTrue(os.path.exists(output_file))
            
            # Cargar datos originales y procesados
            original_df = pd.read_csv(input_file)
            result_df = pd.read_csv(output_file)
            
            # Verificar que los datos están normalizados (valores entre 0 y 1)
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.endswith('_rolling_mean') or col.endswith('_rolling_std'):
                    continue  # Saltar columnas de ventanas deslizantes
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                self.assertGreaterEqual(min_val, -0.1, f"Columna {col} no está normalizada correctamente")
                self.assertLessEqual(max_val, 1.1, f"Columna {col} no está normalizada correctamente")
            
            print(f"✓ Financial data: {len(result_df)} filas procesadas, datos normalizados")
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_irregular_timestamp_data(self):
        """Test con datos de timestamps irregulares"""
        input_file = 'test_data/irregular_timestamp_data.csv'
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            output_file = tmp.name
            
        try:
            # Procesar datos
            self.preprocessor.process(input_file, output_file)
            
            # Verificar que se generó el archivo
            self.assertTrue(os.path.exists(output_file))
            
            # Cargar resultados
            result_df = pd.read_csv(output_file)
            
            # Verificar que hay columnas de ventanas deslizantes
            rolling_cols = [col for col in result_df.columns if 'rolling' in col]
            self.assertGreater(len(rolling_cols), 0, "No se generaron columnas de ventanas deslizantes")
            
            print(f"✓ Irregular timestamp data: {len(result_df)} filas procesadas, {len(rolling_cols)} columnas rolling")
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_mixed_data_types(self):
        """Test con datos de tipos mixtos"""
        input_file = 'test_data/mixed_data_types.csv'
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
            output_file = tmp.name
            
        try:
            # Procesar datos
            self.preprocessor.process(input_file, output_file)
            
            # Verificar que se generó el archivo
            self.assertTrue(os.path.exists(output_file))
            
            # Cargar resultados
            result_df = pd.read_csv(output_file)
            
            # Verificar que el archivo no está vacío
            self.assertGreater(len(result_df), 0, "El archivo procesado está vacío")
            
            # Verificar que hay al menos algunas columnas numéricas
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            self.assertGreater(len(numeric_cols), 0, "No hay columnas numéricas en el resultado")
            
            print(f"✓ Mixed data types: {len(result_df)} filas procesadas, {len(numeric_cols)} columnas numéricas")
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_feature_selection(self):
        """Test de selección de características"""
        # Crear datos de prueba simples
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'unwanted': ['noise'] * 100
        })
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as input_tmp:
            input_file = input_tmp.name
            test_data.to_csv(input_file, index=False)
            
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as output_tmp:
            output_file = output_tmp.name
            
        try:
            # Configurar para incluir solo ciertas características
            config = self.test_config.copy()
            config['preprocessing']['features'] = {
                'include': ['timestamp', 'feature1', 'feature2']
            }
            
            preprocessor = TimeSeriesPreprocessor(config)
            preprocessor.process(input_file, output_file)
            
            # Verificar que solo se incluyen las características especificadas
            result_df = pd.read_csv(output_file)
            expected_cols = {'timestamp', 'feature1', 'feature2'}
            
            # Las columnas rolling se añaden automáticamente
            actual_base_cols = {col for col in result_df.columns 
                              if not col.endswith('_rolling_mean') 
                              and not col.endswith('_rolling_std')}
            
            self.assertEqual(actual_base_cols, expected_cols, 
                           "Las características seleccionadas no coinciden")
            
            print(f"✓ Feature selection: {len(result_df.columns)} columnas en resultado")
            
        finally:
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)
                    
    def test_missing_values_strategies(self):
        """Test de diferentes estrategias para valores faltantes"""
        # Crear datos con valores nulos
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'value': [1, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10]
        })
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as input_tmp:
            input_file = input_tmp.name
            test_data.to_csv(input_file, index=False)
            
        # Test estrategia "drop"
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as output_tmp:
            output_file = output_tmp.name
            
        try:
            config = self.test_config.copy()
            config['preprocessing']['missing_values'] = {
                'strategy': 'drop',
                'threshold': 1.0
            }
            
            preprocessor = TimeSeriesPreprocessor(config)
            preprocessor.process(input_file, output_file)
            
            result_df = pd.read_csv(output_file)
            
            # Verificar que se eliminaron filas con valores nulos
            self.assertLess(len(result_df), len(test_data), 
                          "No se eliminaron filas con valores nulos")
            
            print(f"✓ Missing values drop: {len(result_df)} filas restantes de {len(test_data)} originales")
            
        finally:
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)


class TestConfigurationValidation(unittest.TestCase):
    """Tests para validación de configuración"""
    
    def test_config_loading(self):
        """Test de carga de configuración desde archivo YAML"""
        test_config = {
            'preprocessing': {
                'missing_values': {'strategy': 'fill'},
                'outliers': {'enabled': False}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as tmp:
            yaml.dump(test_config, tmp, default_flow_style=False)
            config_file = tmp.name
            
        try:
            from preprocess import load_config
            loaded_config = load_config(config_file)
            
            self.assertEqual(loaded_config['preprocessing']['missing_values']['strategy'], 'fill')
            self.assertEqual(loaded_config['preprocessing']['outliers']['enabled'], False)
            
            print("✓ Configuration loading: YAML cargado correctamente")
            
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)


def run_integration_tests():
    """Ejecuta tests de integración con los archivos reales"""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS DE INTEGRACIÓN")
    print("="*60)
    
    # Verificar que existen los archivos de prueba
    test_files = [
        'test_data/sensor_data_with_nulls.csv',
        'test_data/financial_data_with_outliers.csv',
        'test_data/irregular_timestamp_data.csv',
        'test_data/mixed_data_types.csv'
    ]
    
    for file in test_files:
        if not os.path.exists(file):
            print(f"❌ Archivo de prueba no encontrado: {file}")
            return False
            
    # Crear directorio de salida para tests
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    try:
        from preprocess import load_config, TimeSeriesPreprocessor
        
        # Cargar configuración por defecto
        config = load_config('config.yaml')
        preprocessor = TimeSeriesPreprocessor(config)
        
        # Procesar cada archivo de prueba
        for test_file in test_files:
            file_name = Path(test_file).stem
            output_file = output_dir / f"processed_{file_name}.csv"
            
            print(f"\nProcesando {test_file}...")
            
            try:
                preprocessor.process(test_file, str(output_file))
                
                # Verificar el resultado
                if output_file.exists():
                    result_df = pd.read_csv(output_file)
                    print(f"✓ {file_name}: {len(result_df)} filas procesadas")
                else:
                    print(f"❌ {file_name}: No se generó archivo de salida")
                    return False
                    
            except Exception as e:
                print(f"❌ {file_name}: Error durante procesamiento - {e}")
                return False
                
        print(f"\n✅ Todos los tests de integración completados exitosamente")
        print(f"📁 Archivos de salida en: {output_dir.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Error en tests de integración: {e}")
        return False


if __name__ == '__main__':
    print("TESTS UNITARIOS DEL PREPROCESADOR DE SERIES TEMPORALES")
    print("="*60)
    
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Ejecutar tests unitarios
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Ejecutar tests de integración
    run_integration_tests()
