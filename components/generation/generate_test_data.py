#!/usr/bin/env python3
"""
Generador de datos de prueba para el preprocesador de series temporales
Crea 4 archivos CSV con diferentes problemas para testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sensor_data_with_nulls():
    """Genera datos de sensores IoT con valores nulos"""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(minutes=15*i) for i in range(2000)]
    
    # Simular datos de sensores
    temperature = [20 + 5*np.sin(i*0.01) + np.random.normal(0, 1) for i in range(2000)]
    humidity = [60 + 10*np.sin(i*0.008) + np.random.normal(0, 2) for i in range(2000)]
    pressure = [1013 + 3*np.sin(i*0.005) + np.random.normal(0, 0.5) for i in range(2000)]
    
    # Introducir valores nulos aleatorios (15% de los datos)
    null_indices = random.sample(range(2000), int(2000 * 0.15))
    for idx in null_indices:
        if random.random() < 0.4:
            temperature[idx] = np.nan
        if random.random() < 0.3:
            humidity[idx] = np.nan
        if random.random() < 0.2:
            pressure[idx] = np.nan
    
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'sensor_id': ['SENSOR_001'] * 2000,
        'location': ['Building_A'] * 2000
    })
    
    return df


def generate_financial_data_with_outliers():
    """Genera datos financieros con outliers extremos"""
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(1500)]
    
    # Simular precio de un activo financiero
    price = 100
    prices = [price]
    volumes = []
    
    for i in range(1, 1500):
        # Movimiento normal del precio
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
        
        # Volumen correlacionado inversamente con el precio
        volume = max(1000, int(50000 - price*100 + np.random.normal(0, 5000)))
        volumes.append(volume)
    
    # El primer volumen
    volumes.insert(0, 45000)
    
    # Introducir outliers extremos (3% de los datos)
    outlier_indices = random.sample(range(1500), int(1500 * 0.03))
    for idx in outlier_indices:
        if random.random() < 0.5:
            # Outlier en precio (caída o subida extrema)
            prices[idx] = prices[idx] * (0.5 if random.random() < 0.5 else 2.0)
        else:
            # Outlier en volumen
            volumes[idx] = volumes[idx] * random.uniform(10, 50)
    
    df = pd.DataFrame({
        'datetime': dates,
        'price': prices,
        'volume': volumes,
        'market_cap': [p * v / 1000 for p, v in zip(prices, volumes)],
        'symbol': ['ACME'] * 1500
    })
    
    return df


def generate_irregular_timestamp_data():
    """Genera datos con timestamps irregulares y gaps"""
    dates = []
    values = []
    
    current_date = datetime(2024, 1, 1)
    
    for i in range(800):
        # Intervalos irregulares (a veces saltos grandes)
        if random.random() < 0.05:  # 5% de probabilidad de gap grande
            jump = timedelta(hours=random.randint(24, 72))
        elif random.random() < 0.2:  # 20% de probabilidad de gap pequeño
            jump = timedelta(minutes=random.randint(60, 300))
        else:  # Intervalo normal
            jump = timedelta(minutes=random.randint(5, 30))
            
        current_date += jump
        dates.append(current_date)
        
        # Valores con tendencia y estacionalidad
        trend = i * 0.01
        seasonal = 10 * np.sin(i * 0.1)
        noise = np.random.normal(0, 2)
        values.append(50 + trend + seasonal + noise)
    
    # Duplicar algunos timestamps para crear problema de duplicados
    duplicate_indices = random.sample(range(len(dates)), 20)
    for idx in duplicate_indices:
        dates.insert(idx + 1, dates[idx])
        values.insert(idx + 1, values[idx] + np.random.normal(0, 0.1))
    
    df = pd.DataFrame({
        'time': dates,
        'measurement': values,
        'device_id': [f'DEVICE_{random.randint(1,5):03d}' for _ in range(len(dates))],
        'status': [random.choice(['active', 'idle', 'maintenance']) for _ in range(len(dates))]
    })
    
    return df


def generate_mixed_data_types():
    """Genera datos con tipos mixtos y problemas de formato"""
    dates = pd.date_range('2024-01-01', periods=1200, freq='30min')
    
    # Mezclar formatos de fecha como strings
    date_strings = []
    for i, date in enumerate(dates):
        if i % 4 == 0:
            date_strings.append(date.strftime('%Y-%m-%d %H:%M:%S'))
        elif i % 4 == 1:
            date_strings.append(date.strftime('%d/%m/%Y %H:%M'))
        elif i % 4 == 2:
            date_strings.append(date.strftime('%Y-%m-%dT%H:%M:%S'))
        else:
            date_strings.append(str(int(date.timestamp())))  # Unix timestamp
    
    # Datos numéricos con algunos como strings
    numeric_data = []
    for i in range(1200):
        value = 75 + 15 * np.sin(i * 0.01) + np.random.normal(0, 3)
        if i % 50 == 0:  # 2% como strings
            numeric_data.append(f"{value:.2f}")
        elif i % 75 == 0:  # Algunos con unidades
            numeric_data.append(f"{value:.1f}°C")
        else:
            numeric_data.append(value)
    
    # Categorías con inconsistencias
    categories = []
    for i in range(1200):
        if i % 100 == 0:
            categories.append(random.choice(['Type A', 'type_a', 'TYPE_A', 'typeA']))
        elif i % 100 == 1:
            categories.append(random.choice(['Type B', 'type_b', 'TYPE_B', 'typeB']))
        else:
            categories.append(random.choice(['Type_C', 'type_c', 'TYPE_C']))
    
    # Valores extremos mezclados
    extreme_values = []
    for i in range(1200):
        if i % 200 == 0:
            extreme_values.append(-999)  # Valor de error común
        elif i % 300 == 0:
            extreme_values.append(999999)  # Valor extremo
        else:
            extreme_values.append(np.random.uniform(0, 100))
    
    df = pd.DataFrame({
        'timestamp': date_strings,
        'temperature': numeric_data,
        'category': categories,
        'reading': extreme_values,
        'quality_flag': [random.choice([0, 1, 2, -1, 'good', 'bad']) for _ in range(1200)],
        'notes': [f'Reading #{i}' if i % 10 == 0 else '' for i in range(1200)]
    })
    
    return df


def main():
    """Genera todos los archivos de prueba"""
    print("Generando archivos de prueba...")
    
    # 1. Datos con valores nulos
    df1 = generate_sensor_data_with_nulls()
    df1.to_csv('/home/gsus/Documents/repos/SyntheticDataSuite/components/generation/test_data/sensor_data_with_nulls.csv', index=False)
    print(f"✓ sensor_data_with_nulls.csv: {df1.shape[0]} filas, {df1.isnull().sum().sum()} valores nulos")
    
    # 2. Datos con outliers
    df2 = generate_financial_data_with_outliers()
    df2.to_csv('/home/gsus/Documents/repos/SyntheticDataSuite/components/generation/test_data/financial_data_with_outliers.csv', index=False)
    print(f"✓ financial_data_with_outliers.csv: {df2.shape[0]} filas con outliers extremos")
    
    # 3. Datos con timestamps irregulares
    df3 = generate_irregular_timestamp_data()
    df3.to_csv('/home/gsus/Documents/repos/SyntheticDataSuite/components/generation/test_data/irregular_timestamp_data.csv', index=False)
    print(f"✓ irregular_timestamp_data.csv: {df3.shape[0]} filas con timestamps irregulares")
    
    # 4. Datos con tipos mixtos
    df4 = generate_mixed_data_types()
    df4.to_csv('/home/gsus/Documents/repos/SyntheticDataSuite/components/generation/test_data/mixed_data_types.csv', index=False)
    print(f"✓ mixed_data_types.csv: {df4.shape[0]} filas con tipos de datos mixtos")
    
    print("\n¡Todos los archivos de prueba generados exitosamente!")


if __name__ == "__main__":
    main()
