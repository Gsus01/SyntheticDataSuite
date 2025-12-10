# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SDL_VIDEODRIVER=dummy \
    PYBULLET_USE_GUI=0

WORKDIR /app

# System dependencies for PyBullet rendering, video logging, and common RL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/inputs /data/outputs /data/config

# Install Python dependencies (pinned for reproducibility)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.2.2 \
        stable-baselines3==2.3.2

# Fetch and install the library from GitHub
RUN git clone --depth 1 --branch main https://github.com/utiasDSL/gym-pybullet-drones.git /app/src \
    && pip install --no-cache-dir /app/src

# Copy the component script and a sample config (overrides the one in the repo)
COPY rl_example.py /app/rl_example.py
COPY variables.json /app/variables.json

# Standard-directory defaults are embedded in rl_example.py, so we keep the command clean
CMD ["python", "/app/rl_example.py"]
