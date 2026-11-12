# Compiles native ARM64 c2patool inside Docker on DGX Spark.
FROM rust:1-bookworm AS c2pa-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev build-essential cmake clang && rm -rf /var/lib/apt/lists/*
RUN cargo install --locked c2patool --version 0.28.2

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates libssl3 libgcc-s1 && rm -rf /var/lib/apt/lists/*
COPY --from=c2pa-builder /usr/local/cargo/bin/c2patool /usr/local/bin/c2patool
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
RUN mkdir -p /data
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
