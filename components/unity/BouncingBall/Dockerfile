FROM ubuntu:22.04

# Evita preguntas durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglu1-mesa \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY BouncingBallSimulation.x86_64 /app/BouncingBallSimulation.x86_64
COPY BouncingBallSimulation_Data /app/BouncingBallSimulation_Data
COPY UnityPlayer.so /app/UnityPlayer.so 
# (Nota: Si Unity no te generó UnityPlayer.so, borra la línea de arriba)
COPY run_simulation.py /app/run_simulation.py

# 4. Dar permisos de ejecución al archivo principal
RUN chmod +x /app/BouncingBallSimulation.x86_64

# 5. Comando por defecto al iniciar
# El wrapper lee un JSON de config y lanza el binario en modo headless
ENTRYPOINT ["python3", "/app/run_simulation.py"]
