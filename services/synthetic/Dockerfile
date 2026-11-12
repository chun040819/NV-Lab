ARG PYTORCH_BASE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${PYTORCH_BASE}
ARG STALL_COMMIT=f6dbb617a08d522f42ac02341028da8c58e669d1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libglib2.0-0 libgl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Obtain STALL and DINOv3 under their OWN licenses; do not bundle model weights.
RUN git clone --no-checkout https://github.com/OmerBenHayun/STALL.git /opt/stall \
    && git -C /opt/stall fetch --depth 1 origin ${STALL_COMMIT} \
    && git -C /opt/stall checkout --detach ${STALL_COMMIT} \
    && git clone --depth 1 https://github.com/facebookresearch/dinov3.git /opt/dinov3
# The NGC image supplies PyTorch/CUDA. Avoid changing torch wheels.
RUN python -m pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.2 \
    opencv-python-headless==4.10.0.84 Pillow numpy
WORKDIR /service
COPY app.py /service/app.py
ENV STALL_DIR=/opt/stall \
    DINO_V3_REPO_DIR=/opt/dinov3 \
    DINO_V3_WEIGHTS=/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    STALL_PARAMS=/opt/stall/precomputed/stall_params_vatex_dino_v3.npz \
    STALL_MAX_WINDOWS=3 STALL_FRAME_BATCH=4
EXPOSE 8093
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8093","--workers","1","--no-access-log"]
