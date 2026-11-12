ARG PYTORCH_BASE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${PYTORCH_BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir 'fastapi==0.115.12' 'uvicorn==0.34.2' 'numpy<3'
# Original AASIST MIT source and published pretrained model. No training is performed.
RUN git clone --depth 1 https://github.com/clovaai/aasist.git /opt/aasist && \
    cd /opt/aasist && git lfs pull && \
    test -s /opt/aasist/models/weights/AASIST.pth
WORKDIR /srv
COPY app.py /srv/app.py
EXPOSE 8092
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
