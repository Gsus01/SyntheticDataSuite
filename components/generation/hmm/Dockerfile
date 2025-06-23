FROM python:3.12-slim


# Docker cacheará esta capa si requirements.txt no cambia
COPY requirements.txt .

# Instalar dependencias listadas en requirements.txt
# --no-cache-dir ayuda a reducir el tamaño final de la imagen
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . /app



ENTRYPOINT ["python","generar_datos.py"]

