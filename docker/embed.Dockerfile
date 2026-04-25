# Embedding pipeline image — SPECTER2 + aggregation + kNN + UMAP.
# Base: pytorch official with CUDA 12.4, works under CUDA 13 drivers.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HF_HUB_CACHE=/cache/hf/hub \
    TOKENIZERS_PARALLELISM=false

RUN pip install --no-cache-dir \
    'transformers>=4.40,<4.50' \
    'sentence-transformers>=3.0,<4' \
    'pandas>=2.0' \
    'pyarrow>=14' \
    'duckdb>=1.0' \
    'umap-learn>=0.5' \
    'scikit-learn>=1.3' \
    'scipy>=1.11' \
    'tqdm>=4.66' \
    'networkx>=3.2' \
    'python-igraph>=0.11' \
    'leidenalg>=0.10' \
    'pyyaml>=6' \
    'unidecode>=1.3' \
    'rapidfuzz>=3.6' \
    'matplotlib>=3.7' \
    'python-dotenv>=1.0' \
    'optimum[onnxruntime]>=1.20' \
    'huggingface_hub>=0.24'

WORKDIR /work
