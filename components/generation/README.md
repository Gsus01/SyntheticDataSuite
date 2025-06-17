# Preprocesador de Series Temporales

Microservicio de preprocesamiento de datos para series temporales, diseñado para ejecutarse de forma aislada en workflows de Argo. Permite el procesamiento configurable de archivos CSV con múltiples opciones de preprocesamiento específicas para datos temporales.

## 🚀 Características

- **Preprocesamiento configurable** mediante archivos YAML
- **Detección automática** de columnas temporales
- **Manejo de valores faltantes** con múltiples estrategias
- **Detección y manejo de outliers** (IQR y Z-score)
- **Normalización de datos** (MinMax, Z-score, Robust)
- **Remuestreo temporal** para datos irregulares
- **Ventanas deslizantes** para análisis temporal
- **Selección de características** configurable
- **Ejecución en Docker** para aislamiento completo
- **Tests exhaustivos** con datos de prueba

## 📋 Requisitos

- Python 3.11+
- Docker (opcional, para ejecución aislada)
- Dependencias: pandas, numpy, PyYAML

## 🛠️ Instalación

### Opción 1: Ejecución Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Generar datos de prueba
python generate_test_data.py
```

### Opción 2: Docker

```bash
# Construir imagen
docker build -t timeseries-preprocessor .

# O usar el script de utilidad
./run.sh build
```

## 📖 Uso

### Ejecución Básica

```bash
python preprocess.py --config config.yaml --input data.csv --output processed_data.csv
```

### Con Docker

```bash
docker run --rm \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/data.csv:/app/data/input/data.csv:ro \
  -v $(pwd)/output:/app/data/output \
  timeseries-preprocessor \
  --config /app/config.yaml \
  --input /app/data/input/data.csv \
  --output /app/data/output/processed.csv
```

### Con Docker Compose

```bash
# Procesamiento único
docker-compose up timeseries-preprocessor

# Procesamiento en lote
docker-compose --profile batch up batch-processor
```

### Script de Utilidad

```bash
# Ejecutar demo completo
./run.sh demo

# Ejecutar tests
./run.sh test

# Procesar archivo específico
./run.sh run --input data.csv --output processed.csv

# Procesar con Docker
./run.sh run --input data.csv --output processed.csv --docker
```

## ⚙️ Configuración

El procesamiento se configura mediante archivos YAML. Ejemplo de configuración:

```yaml
# Configuración de logging
logging:
  level: INFO

# Configuración de datos
data:
  datetime_column: null  # Auto-detección si es null

# Pipeline de preprocesamiento
preprocessing:
  # Selección de características
  features:
    include: ["timestamp", "value", "sensor_1"]
    exclude: ["id", "metadata"]
    
  # Manejo de valores faltantes
  missing_values:
    strategy: "fill"  # "drop" o "fill"
    method: "interpolate"  # "forward", "backward", "interpolate", "mean"
    threshold: 0.5  # Para strategy="drop"
    
  # Detección de outliers
  outliers:
    enabled: true
    method: "iqr"  # "iqr" o "zscore"
    threshold: 3   # Para método zscore
    action: "cap"  # "remove" o "cap"
    
  # Normalización
  normalization:
    enabled: true
    method: "minmax"  # "minmax", "zscore", "robust"
    
  # Remuestreo temporal
  resampling:
    enabled: false
    frequency: "1H"  # "1min", "5min", "1H", "1D"
    method: "mean"   # "mean", "sum", "first", "last"
    
  # Ventanas deslizantes
  sliding_windows:
    enabled: true
    size: 5
    features: []  # Vacío = todas las numéricas
```

## 📁 Estructura del Proyecto

```
components/generation/
├── preprocess.py              # Módulo principal del preprocesador
├── config.yaml               # Configuración por defecto
├── batch_config.yaml         # Configuración para lote
├── requirements.txt          # Dependencias Python
├── Dockerfile               # Imagen Docker
├── docker-compose.yml       # Orquestación de contenedores
├── run.sh                   # Script de utilidad
├── generate_test_data.py    # Generador de datos de prueba
├── test_preprocessor.py     # Tests unitarios e integración
├── test_data/              # Datos de prueba
│   ├── sensor_data_with_nulls.csv
│   ├── financial_data_with_outliers.csv
│   ├── irregular_timestamp_data.csv
│   └── mixed_data_types.csv
└── README.md               # Este archivo
```

## 🧪 Tests

El proyecto incluye tests exhaustivos con 4 archivos de datos de prueba, cada uno con problemas específicos:

### Datos de Prueba

1. **sensor_data_with_nulls.csv**: Datos de sensores IoT con 15% de valores nulos
2. **financial_data_with_outliers.csv**: Datos financieros con outliers extremos (3%)
3. **irregular_timestamp_data.csv**: Timestamps irregulares con gaps y duplicados
4. **mixed_data_types.csv**: Tipos de datos mixtos y formatos inconsistentes

### Ejecución de Tests

```bash
# Tests unitarios e integración
python test_preprocessor.py

# O con el script de utilidad
./run.sh test

# Demo con todos los archivos
./run.sh demo
```

## 🐳 Docker

### Dockerfile

El contenedor está optimizado para:
- **Tamaño mínimo**: Base Python 3.11-slim
- **Seguridad**: Usuario no-root
- **Flexibilidad**: Volúmenes montables para datos y configuración
- **Logging**: Output sin buffer para mejor observabilidad

### Docker Compose

Incluye dos servicios:
- `timeseries-preprocessor`: Procesamiento único
- `batch-processor`: Procesamiento en lote (profile: batch)

## 🔧 Integración con Argo Workflows

Ejemplo de uso en Argo Workflows:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: preprocess-timeseries
spec:
  entrypoint: preprocess-data
  templates:
  - name: preprocess-data
    container:
      image: timeseries-preprocessor:latest
      command: ["python", "preprocess.py"]
      args: [
        "--config", "/config/config.yaml",
        "--input", "/data/input/raw_data.csv",
        "--output", "/data/output/processed_data.csv"
      ]
      volumeMounts:
      - name: config
        mountPath: /config
      - name: data
        mountPath: /data
  volumes:
  - name: config
    configMap:
      name: preprocessing-config
  - name: data
    persistentVolumeClaim:
      claimName: data-pvc
```

## 📊 Funcionalidades de Preprocesamiento

### Detección de Columnas Temporales
- Auto-detección por nombre (date, time, timestamp)
- Auto-detección por contenido
- Conversión automática a datetime
- Indexación temporal

### Manejo de Valores Faltantes
- **Drop**: Eliminar filas/columnas con umbrales configurables
- **Fill**: Rellenar con forward/backward fill, interpolación o media

### Detección de Outliers
- **IQR Method**: Rango intercuartílico (Q1-1.5*IQR, Q3+1.5*IQR)
- **Z-Score**: Desviación estándar configurable
- **Acciones**: Eliminar o limitar valores

### Normalización
- **MinMax**: Escalar a [0,1]
- **Z-Score**: Media 0, desviación 1
- **Robust**: Mediana y rango intercuartílico

### Remuestreo Temporal
- Frecuencias configurables (minutos, horas, días)
- Métodos de agregación (media, suma, primero, último)
- Manejo automático de índices temporales

### Ventanas Deslizantes
- Media móvil y desviación estándar
- Tamaño de ventana configurable
- Aplicable a columnas específicas o todas las numéricas

## 🚨 Manejo de Errores

El preprocesador incluye manejo robusto de errores:
- Validación de archivos de entrada
- Logging detallado de operaciones
- Manejo de tipos de datos inconsistentes
- Recuperación de errores de parsing temporal

## 📈 Logging y Monitoreo

- Logging estructurado con timestamps
- Niveles configurables (DEBUG, INFO, WARNING, ERROR)
- Métricas de procesamiento (filas originales vs finales)
- Output compatible con sistemas de monitoreo

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

Para reportar problemas o solicitar funcionalidades:
1. Crear un issue en el repositorio
2. Incluir logs de error y configuración utilizada
3. Proporcionar datos de ejemplo si es posible
