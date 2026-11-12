ARG PYTORCH_BASE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${PYTORCH_BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
# Never overwrite CUDA-enabled torch/torchvision baked into NVIDIA's container.
# DeepGuard 1.1.1 is deliberately --no-deps; dependency versions are validated at startup.
RUN python -m pip install --no-cache-dir \
  'fastapi==0.115.12' 'uvicorn==0.34.2' \
  'timm>=1.0.12,<1.1' 'ultralytics>=8.3.0,<9' \
  'albumentations>=2.0,<3' 'torchmetrics>=1.6,<2' \
  'opencv-python-headless>=4.10,<5' 'pandas>=2.2,<3' \
  'scikit-learn>=1.6,<2' 'grad-cam>=1.5,<2' \
  'colorama>=0.4,<1'
RUN python -m pip install --no-cache-dir --no-deps 'deepguard==1.1.1'
WORKDIR /srv
COPY app.py /srv/app.py
EXPOSE 8091
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
