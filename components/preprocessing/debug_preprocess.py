#!/usr/bin/env python3
"""
Script de debug para identificar el error NoneType
"""

from preprocess import load_config, TimeSeriesPreprocessor

def debug_preprocessing():
    """Debug del preprocesamiento paso a paso"""
    print("=== DEBUG DEL PREPROCESAMIENTO ===")
    
    # Cargar configuración
    print("1. Cargando configuración...")
    config = load_config('config.yaml')
    print(f"Config cargado: {type(config)}")
    print(f"Features config: {config.get('preprocessing', {}).get('features', {})}")
    
    # Crear preprocesador
    print("2. Creando preprocesador...")
    preprocessor = TimeSeriesPreprocessor(config)
    
    # Cargar datos
    print("3. Cargando datos...")
    test_file = 'test_data/sensor_data_with_nulls.csv'
    df = preprocessor.load_data(test_file)
    print(f"Datos cargados: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Probar cada paso del pipeline
    try:
        print("4. Preprocesando datetime...")
        df = preprocessor.preprocess_datetime(df)
        print("✓ DateTime OK")
    except Exception as e:
        print(f"❌ DateTime Error: {e}")
        return
    
    try:
        print("5. Seleccionando features...")
        df = preprocessor.select_features(df)
        print("✓ Features OK")
    except Exception as e:
        print(f"❌ Features Error: {e}")
        return
    
    try:
        print("6. Manejando valores faltantes...")
        df = preprocessor.handle_missing_values(df)
        print("✓ Missing values OK")
    except Exception as e:
        print(f"❌ Missing values Error: {e}")
        return
    
    try:
        print("7. Detectando outliers...")
        df = preprocessor.detect_outliers(df)
        print("✓ Outliers OK")
    except Exception as e:
        print(f"❌ Outliers Error: {e}")
        return
    
    try:
        print("8. Normalizando datos...")
        df = preprocessor.normalize_data(df)
        print("✓ Normalization OK")
    except Exception as e:
        print(f"❌ Normalization Error: {e}")
        return
    
    try:
        print("9. Remuestreando datos...")
        df = preprocessor.resample_data(df)
        print("✓ Resampling OK")
    except Exception as e:
        print(f"❌ Resampling Error: {e}")
        return
    
    try:
        print("10. Creando ventanas deslizantes...")
        df = preprocessor.create_sliding_windows(df)
        print("✓ Sliding windows OK")
    except Exception as e:
        print(f"❌ Sliding windows Error: {e}")
        return
    
    print("✅ Todo el pipeline completado exitosamente!")

if __name__ == "__main__":
    debug_preprocessing()
