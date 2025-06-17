# 🚀 Preprocesador de Series Temporales - Docker Compose

Este directorio contiene un preprocesador de series temporales completamente dockerizado que puede ejecutarse fácilmente usando Docker Compose.

## 📋 Servicios Disponibles

### 🔬 Test Rápido
```bash
docker compose run --rm test-preprocessor
```
Procesa datos de sensores con configuración por defecto para testing rápido.

### 🌡️ Procesar Datos de Sensores
```bash
docker compose run --rm process-sensors
```
Procesa `sensor_data_with_nulls.csv` con el pipeline completo de preprocesamiento.

### 💰 Procesar Datos Financieros
```bash
docker compose run --rm process-financial
```
Procesa `financial_data_with_outliers.csv` aplicando detección de outliers y normalización.

### ⏰ Procesar Datos con Timestamps Irregulares
```bash
docker compose run --rm process-irregular
```
Procesa `irregular_timestamp_data.csv` manejando timestamps no uniformes.

### 🔀 Procesar Datos de Tipos Mixtos
```bash
docker compose run --rm process-mixed
```
Procesa `mixed_data_types.csv` con diferentes tipos de datos.

### 📦 Procesamiento en Lote
```bash
docker compose run --rm batch-processor
```
Procesa múltiples archivos usando configuración batch.

## 🎯 Comandos Personalizados

Puedes usar cualquier servicio con argumentos personalizados:

```bash
# Procesar archivo específico con salida personalizada
docker compose run --rm test-preprocessor \
  --config /app/config.yaml \
  --input /app/data/input/tu_archivo.csv \
  --output /app/data/output/tu_salida.csv

# Usar configuración personalizada
docker compose run --rm test-preprocessor \
  --config /app/tu_config.yaml \
  --input /app/data/input/datos.csv \
  --output /app/data/output/resultado.csv
```

## 📁 Estructura de Directorios

```
.
├── test_data/              # Datos de entrada (montado como solo lectura)
│   ├── sensor_data_with_nulls.csv
│   ├── financial_data_with_outliers.csv
│   ├── irregular_timestamp_data.csv
│   └── mixed_data_types.csv
├── output/                 # Resultados procesados
├── config.yaml            # Configuración principal
├── batch_config.yaml      # Configuración para procesamiento batch
├── Dockerfile             # Definición de la imagen Docker
├── docker-compose.yml     # Definición de servicios
└── run_help.sh           # Script de ayuda con ejemplos
```

## 🔧 Pipeline de Preprocesamiento

El sistema aplica automáticamente:

1. **Detección de columnas temporales** - Identifica automáticamente timestamps
2. **Selección de características** - Incluye/excluye columnas según configuración
3. **Manejo de valores faltantes** - Interpolación, forward-fill, o eliminación
4. **Detección de outliers** - Métodos IQR o Z-score con acción configurable
5. **Normalización** - MinMax, Z-score, o Robust scaling
6. **Remuestreo temporal** - Agregación por intervalos de tiempo
7. **Ventanas deslizantes** - Media y desviación estándar móviles

## ⚙️ Configuración

La configuración se define en `config.yaml`:

```yaml
logging:
  level: INFO

data:
  datetime_column: null  # Detección automática

preprocessing:
  missing_values:
    strategy: "fill"      # "drop" o "fill"
    method: "interpolate" # "forward", "backward", "interpolate", "mean"
    
  outliers:
    enabled: true
    method: "iqr"         # "iqr" o "zscore"
    action: "cap"         # "remove" o "cap"
    
  normalization:
    enabled: true
    method: "minmax"      # "minmax", "zscore", "robust"
    
  sliding_windows:
    enabled: true
    size: 5
```

## 🚀 Inicio Rápido

1. **Ver ayuda completa:**
   ```bash
   ./run_help.sh
   ```

2. **Test rápido:**
   ```bash
   docker compose run --rm test-preprocessor
   ```

3. **Verificar resultados:**
   ```bash
   ls -la output/
   head -5 output/test_processed_data.csv
   ```

4. **Limpiar archivos de salida:**
   ```bash
   rm -rf output/* && mkdir -p output/
   ```

## 📊 Ejemplos de Resultados

### Entrada (sensor_data_with_nulls.csv):
```csv
timestamp,temperature,humidity,pressure,sensor_id,location
2024-01-01 00:00:00,25.3,60.2,1013.25,SENSOR_001,Building_A
2024-01-01 00:15:00,,58.7,1012.80,SENSOR_001,Building_A
```

### Salida (processed_sensor_data.csv):
```csv
timestamp,temperature,humidity,pressure,sensor_id,location,temperature_rolling_mean,temperature_rolling_std,humidity_rolling_mean,humidity_rolling_std,pressure_rolling_mean,pressure_rolling_std
2024-01-01 00:00:00,0.461,0.515,0.617,SENSOR_001,Building_A,,,,,,
2024-01-01 00:15:00,0.572,0.459,0.588,SENSOR_001,Building_A,,,,,,
```

## 🔍 Monitoreo y Logs

Los logs muestran el progreso detallado:
- ✅ Datos cargados y dimensiones
- ✅ Columna temporal detectada
- ✅ Valores nulos procesados
- ✅ Outliers detectados y manejados
- ✅ Datos normalizados
- ✅ Ventanas deslizantes creadas
- ✅ Forma final y ubicación del archivo

## 🛠️ Desarrollo

Para modificar la configuración o el código:

1. **Editar configuración:**
   ```bash
   nano config.yaml
   ```

2. **Reconstruir imagen:**
   ```bash
   docker compose build
   ```

3. **Probar cambios:**
   ```bash
   docker compose run --rm test-preprocessor
   ```

## 🐛 Troubleshooting

### Error de permisos
Si encuentras errores de permisos, asegúrate de que el directorio `output/` tenga los permisos correctos:
```bash
rm -rf output/ && mkdir -p output/
```

### Variables de entorno
Modifica `.env` para ajustar UID/GID si es necesario:
```bash
UID=1000
GID=1000
```
