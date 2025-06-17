# Preprocesador de Series Temporales

Microservicio de preprocesamiento de datos para series temporales, diseñado para ejecutarse de forma aislada en workflows de Argo. Permite el procesamiento configurable de archivos CSV con múltiples opciones de preprocesamiento específicas para datos temporales.

## 🚀 Características

- **Preprocesamiento configurable** mediante archivos YAML
- **Detección automática** de columnas temporales (con opción de formato específico)
- **Manejo de valores faltantes** con múltiples estrategias
- **Detección y manejo de outliers** (IQR y Z-score)
- **Normalización de datos** (MinMax, Z-score, Robust)
- **Remuestreo temporal** para datos irregulares
- **Ventanas deslizantes** para análisis temporal
- **Selección de características** configurable
- **Ejecución en Docker** para aislamiento completo
- **Tests exhaustivos** con pytest (unitarios, integración, caja negra)

## 📋 Requisitos

- Python 3.11+
- Docker (opcional, para ejecución aislada)
- Dependencias: pandas, numpy, PyYAML, pytest (ver `requirements.txt`)

## 🛠️ Instalación

### Opción 1: Ejecución Local

```bash
# Instalar dependencias (incluye pytest)
pip install -r requirements.txt
```
Los datos de prueba necesarios se encuentran en el directorio `test_data/`.

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
# Procesamiento único (usando el servicio 'timeseries-preprocessor')
docker-compose up timeseries-preprocessor

# Procesamiento en lote
docker-compose --profile batch up batch-processor
```

### Script de Utilidad

```bash
# Ejecutar demo completo
./run.sh demo

# Ejecutar tests (actualizado para pytest)
./run.sh test

# Procesar archivo específico
./run.sh run --input data.csv --output processed.csv

# Procesar con Docker
./run.sh run --input data.csv --output processed.csv --docker
```

## ⚙️ Configuración

El procesamiento se configura mediante archivos YAML. Ejemplo de configuración (`config.yaml`):

```yaml
# Configuración de logging
logging:
  level: INFO

# Configuración de datos
data:
  datetime_column: null  # Auto-detección si es null
  datetime_format: null  # Opcional, ej: "%Y-%m-%d %H:%M:%S" o "%d/%m/%Y %H:%M". Si es null, se intenta auto-detección.

# Pipeline de preprocesamiento
preprocessing:
  # Selección de características
  features:
    # include: ["timestamp", "value", "sensor_1"]
    # exclude: ["id", "metadata"]
    
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
    frequency: "h"  # Ej: "1min", "5min", "h" (hora), "D" (día)
    method: "mean"   # "mean", "sum", "first", "last"
    
  # Ventanas deslizantes
  sliding_windows:
    enabled: true
    size: 5
    features: []  # Vacío = todas las numéricas
```

## 📁 Estructura del Proyecto

```
components/preprocessing/
├── preprocess.py              # Módulo principal del preprocesador
├── config.yaml               # Configuración por defecto
├── batch_config.yaml         # Configuración para lote
├── requirements.txt          # Dependencias Python (incluye pytest)
├── Dockerfile               # Imagen Docker
├── docker-compose.yml       # Orquestación de contenedores (incluye servicio para black-box tests)
├── run.sh                   # Script de utilidad (actualizado para pytest)
├── test_data/               # Datos de prueba para integración y black-box
│   ├── sensor_data_with_nulls.csv
│   ├── financial_data_with_outliers.csv
│   ├── irregular_timestamp_data.csv
│   └── mixed_data_types.csv
├── tests/                   # Directorio principal de tests
│   ├── __init__.py
│   ├── unit/                # Tests Unitarios
│   │   ├── __init__.py
│   │   └── test_preprocessor_units.py
│   ├── integration/         # Tests de Integración
│   │   ├── __init__.py
│   │   └── test_preprocessor_integration.py
│   └── blackbox/            # Tests de Caja Negra (Black-Box)
│       ├── __init__.py
│       └── test_container_execution.py
└── README.md                # Este archivo
```

## 🧪 Tests

El proyecto utiliza `pytest` para un testing exhaustivo por capas, cubriendo tests unitarios, de integración y de caja negra (black-box).

### Estructura de los Tests

- **Tests Unitarios (`tests/unit/`)**: Prueban funciones y métodos individuales de la clase `TimeSeriesPreprocessor` de forma aislada. Se centran en la lógica interna sin depender de I/O de ficheros reales o Docker.
- **Tests de Integración (`tests/integration/`)**: Verifican que el pipeline completo de preprocesamiento (`process` method) funciona como se espera, incluyendo la lectura de ficheros CSV de prueba (de `test_data/`) y la escritura de resultados. Utilizan configuraciones específicas y la fixture `tmp_path` de pytest para manejar ficheros temporales.
- **Tests de Caja Negra (`tests/blackbox/`)**: Prueban la imagen Docker construida como una unidad cerrada. Simulan cómo se ejecutaría el servicio en un entorno de producción (ej. Argo Workflows). Utilizan `docker-compose` para ejecutar el contenedor con datos y configuraciones montadas, verificando los ficheros de salida. El servicio `blackbox-test-runner` en `docker-compose.yml` está dedicado a esto.

### Datos de Prueba

Los datos de prueba se encuentran en el directorio `test_data/` y son utilizados por los tests de integración y caja negra:

1.  **sensor_data_with_nulls.csv**: Datos de sensores IoT con valores nulos.
2.  **financial_data_with_outliers.csv**: Datos financieros con outliers extremos.
3.  **irregular_timestamp_data.csv**: Timestamps irregulares con gaps y formatos mixtos.
4.  **mixed_data_types.csv**: Tipos de datos mixtos y formatos inconsistentes.

### Ejecución de Tests

Asegúrate de tener `pytest` instalado (incluido en `requirements.txt`):
```bash
# Desde components/preprocessing/
pip install -r requirements.txt
```

**1. Ejecutar todos los tests:**
Desde el directorio `components/preprocessing/`:
```bash
python3 -m pytest tests/
```
O simplemente (si pytest está en el PATH y el entorno reconoce los módulos locales):
```bash
pytest tests/
```

**2. Ejecutar tests de una capa específica:**
Desde el directorio `components/preprocessing/`:
```bash
# Tests Unitarios
python3 -m pytest tests/unit/

# Tests de Integración
python3 -m pytest tests/integration/

# Tests de Caja Negra (requieren Docker y Docker Compose v1 o v2)
# Asegúrate que Docker está corriendo y docker-compose (o docker compose) está en tu PATH
python3 -m pytest tests/blackbox/
```

**3. Usar el script de utilidad `run.sh`:**
El script ha sido actualizado para usar `pytest`.
```bash
# Desde components/preprocessing/
./run.sh test
```

**4. Ver más detalles y logs:**
Usa las opciones de `pytest` para mayor verbosidad (desde `components/preprocessing/`):
```bash
python3 -m pytest -s -v tests/
```
Para los tests de caja negra, los logs del contenedor también pueden ser inspeccionados (la salida de `docker compose run` o `docker-compose run` incluye stdout/stderr del contenedor, que son capturados por pytest).

### Demo
El comando de demo sigue disponible:
```bash
# Desde components/preprocessing/
./run.sh demo
```

## 🐳 Docker

### Dockerfile

El contenedor está optimizado para:
- **Tamaño mínimo**: Base Python 3.11-slim.
- **Seguridad**: Usuario no-root (`preprocessor`).
- **Flexibilidad**: Volúmenes montables para datos y configuración.
- **Logging**: Output sin buffer para mejor observabilidad.
- Incluye `config.yaml` por defecto en `/app/config.yaml`.

### Docker Compose

El fichero `docker-compose.yml` define varios servicios, incluyendo:
- `timeseries-preprocessor`: Para ejecución única del preprocesador.
- `batch-processor`: Para procesamiento en lote (activado con `--profile batch`).
- `blackbox-test-runner`: Servicio base utilizado por los tests de caja negra.

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
      image: timeseries-preprocessor:latest # Asegúrate que esta imagen está disponible en tu registro
      command: ["python", "preprocess.py"]
      args: [
        "--config", "/app/config.yaml", # Puede ser el config.yaml por defecto en la imagen o uno montado
        "--input", "/data/input/raw_data.csv",
        "--output", "/data/output/processed_data.csv"
      ]
      volumeMounts:
      - name: config-volume # Opcional: si quieres montar un configMap como config.yaml
        mountPath: /app/config.yaml # Sobrescribiría el de la imagen
        subPath: config.yaml # Asumiendo que tu configMap tiene una clave config.yaml
      - name: data-input-volume
        mountPath: /data/input # Directorio para datos de entrada
      - name: data-output-volume
        mountPath: /data/output # Directorio para datos de salida
  volumes:
  - name: config-volume
    configMap:
      name: preprocessing-config # Nombre de tu ConfigMap
  - name: data-input-volume
    persistentVolumeClaim: # O cualquier otro tipo de volumen
      claimName: input-data-pvc
  - name: data-output-volume
    persistentVolumeClaim:
      claimName: output-data-pvc
```
Asegúrate de que la imagen `timeseries-preprocessor:latest` esté accesible para tu clúster de Argo y que los volúmenes y ConfigMaps estén configurados correctamente.

## 📊 Funcionalidades de Preprocesamiento

(Esta sección parece estar actualizada y no requiere cambios inmediatos basados en la refactorización de tests)

### Detección de Columnas Temporales
- Auto-detección por nombre (date, time, timestamp)
- Auto-detección por contenido
- Opción para especificar formato de fecha/tiempo (`data.datetime_format`)
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
- Recuperación de errores de parsing temporal (convierte a NaT y puede eliminar filas)

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

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles. (Asumiendo que existe un archivo LICENSE, si no, esta línea podría necesitar ajuste)

## 🆘 Soporte

Para reportar problemas o solicitar funcionalidades:
1. Crear un issue en el repositorio
2. Incluir logs de error y configuración utilizada
3. Proporcionar datos de ejemplo si es posible
