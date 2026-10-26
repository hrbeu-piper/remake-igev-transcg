# Official NVIDIA CUDA 12.8.2 / Ubuntu 22.04, verified linux/amd64 digest.
ARG CUDA_BASE=nvcr.io/nvidia/cuda@sha256:0d3cb6d53388903d4292c575da1576d0c1ef1f54e475892d7ea8d0b38490b4ce
FROM ${CUDA_BASE}

# conda-pack 0.8.1: dest_prefix=/opt/remake-env, arcroot=opt/remake-env.
# The archive is a separately verified release artifact, never a Git file.
ADD env-layer.tar.gz /

WORKDIR /opt/remake-igev
COPY src/ ./src/
COPY vendor/ ./vendor/
COPY runtime-manifest.json ./runtime-manifest.json
COPY scripts/ ./scripts/
COPY README.md THIRD_PARTY_NOTICES.md ./
COPY docs/ ./docs/

ENV PATH="/opt/remake-env/bin:${PATH}" \
    PYTHONPATH="/opt/remake-igev/src" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    OMP_NUM_THREADS="2" \
    OPENBLAS_NUM_THREADS="2" \
    MKL_NUM_THREADS="2"

# No data/weight download, bootstrap, training or receipt reuse at startup.
CMD ["python", "/opt/remake-igev/scripts/healthcheck.py"]
