#!/usr/bin/env bash
# Run the embed pipeline in the GPU-enabled Docker image.
#
#   GPU=1 ./docker/run_embed.sh embed          # run scripts/05_embed_papers.py
#   GPU=1 ./docker/run_embed.sh similarity     # run scripts/06_author_similarity.py
#   GPU=1 ./docker/run_embed.sh shell          # drop into bash inside the container
set -euo pipefail

IMAGE="${IMAGE:-transport-atlas-embed:v1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-/data2/chois/hf-cache}"
EMBED_OUT="${EMBED_OUT:-/data2/chois/transport-atlas}"
GPU="${GPU:-1}"

mkdir -p "$CACHE_DIR" "$EMBED_OUT"

COMMON_ARGS=(
  --rm
  --gpus "\"device=${GPU}\""
  --shm-size=8g
  -v "$REPO_ROOT":/work
  -v "$CACHE_DIR":/cache/hf
  -v "$EMBED_OUT":/embed
  -e PYTHONPATH=/work/src
  -e EMBED_OUT=/embed
  -e HF_HOME=/cache/hf
  -e NUMBA_CACHE_DIR=/tmp/numba-cache
  -e MPLCONFIGDIR=/tmp/mpl-cache
  -w /work
  -u "$(id -u):$(id -g)"
)

CMD="${1:-embed}"
shift || true

case "$CMD" in
  embed)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/05_embed_papers.py "$@"
    ;;
  similarity|sim)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/06_author_similarity.py "$@"
    ;;
  both)
    docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/05_embed_papers.py "$@"
    docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/06_author_similarity.py
    ;;
  shell)
    exec docker run -it "${COMMON_ARGS[@]}" "$IMAGE" bash
    ;;
  *)
    echo "usage: $0 {embed|similarity|both|shell}" >&2
    exit 2
    ;;
esac
