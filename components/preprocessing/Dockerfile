FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py .
COPY config.yaml .
COPY batch_config.yaml .

RUN mkdir -p /app/data/input /app/data/output

RUN useradd -m -u 1000 preprocessor && chown -R preprocessor:preprocessor /app
USER preprocessor

ENTRYPOINT ["python", "preprocess.py"]
CMD ["--config", "config.yaml", "--input", "/app/data/input/data.csv", "--output", "/app/data/output/processed_data.csv"]
