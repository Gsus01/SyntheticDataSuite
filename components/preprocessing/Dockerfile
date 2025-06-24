FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY preprocess.py .
COPY config.yaml .

# Crear directorios para datos
RUN mkdir -p /app/data/input /app/data/output

# Configurar usuario no root para seguridad
RUN useradd -m -u 1000 preprocessor && chown -R preprocessor:preprocessor /app
USER preprocessor

# Punto de entrada
ENTRYPOINT ["python", "preprocess.py"]

# Comando por defecto (se puede sobrescribir)
CMD ["--config", "config.yaml", "--input", "/app/data/input/data.csv", "--output", "/app/data/output/processed_data.csv"]
