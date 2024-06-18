FROM python:3.10

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libssl-dev \
    libffi-dev \
    ffmpeg \
    python3-dev \
    gcc \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
