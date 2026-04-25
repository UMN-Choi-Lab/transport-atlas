#!/usr/bin/env bash
# Run the embed + paper-analysis pipeline in the GPU-enabled Docker image.
#
# Pipeline stages:
#   GPU=1 ./docker/run_embed.sh embed          # scripts/05_embed_papers.py
#   GPU=1 ./docker/run_embed.sh similarity     # scripts/06_author_similarity.py
#   GPU=1 ./docker/run_embed.sh phantom        # scripts/07_phantom_eval.py (§8)
#
# Paper analyses (outputs → paper/manuscript/{tables,figures}):
#   ./docker/run_embed.sh descriptive          # paper/analysis/01_descriptive_tables.py
#   ./docker/run_embed.sh coauthor             # paper/analysis/02_coauthor_structure.py
#   ./docker/run_embed.sh partition            # paper/analysis/03_partition_alignment.py
#   ./docker/run_embed.sh phantom-fig          # paper/analysis/04_phantom_eval.py
#   ./docker/run_embed.sh trajectories         # paper/analysis/05_trajectory_taxonomy.py
#   ./docker/run_embed.sh analysis <path> …    # arbitrary paper/analysis/*.py
#
# Site build:
#   ./docker/run_embed.sh annotate             # scripts/03b_annotate_all_coauthors.py
#   ./docker/run_embed.sh reflag-phantoms      # scripts/06b_reflag_phantoms.py
#   ./docker/run_embed.sh render               # scripts/04_render.py
#
# Misc:
#   ./docker/run_embed.sh shell                # interactive bash in the container
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
  finetune)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/08_finetune_specter2.py "$@"
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
  phantom)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/07_phantom_eval.py "$@"
    ;;
  descriptive)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python paper/analysis/01_descriptive_tables.py "$@"
    ;;
  coauthor)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python paper/analysis/02_coauthor_structure.py "$@"
    ;;
  partition)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python paper/analysis/03_partition_alignment.py "$@"
    ;;
  phantom-fig)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python paper/analysis/04_phantom_eval.py "$@"
    ;;
  trajectories|traj)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python paper/analysis/05_trajectory_taxonomy.py "$@"
    ;;
  annotate)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/03b_annotate_all_coauthors.py "$@"
    ;;
  reflag-phantoms)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/06b_reflag_phantoms.py "$@"
    ;;
  render)
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" \
      python scripts/04_render.py "$@"
    ;;
  analysis)
    # Pass-through for any other paper/analysis script: first arg is the path.
    if [ $# -lt 1 ]; then
      echo "usage: $0 analysis <script.py> [args …]" >&2
      exit 2
    fi
    exec docker run "${COMMON_ARGS[@]}" "$IMAGE" python "$@"
    ;;
  shell)
    exec docker run -it "${COMMON_ARGS[@]}" "$IMAGE" bash
    ;;
  *)
    echo "usage: $0 {embed|finetune|similarity|both|phantom|descriptive|coauthor|partition|phantom-fig|trajectories|annotate|reflag-phantoms|render|analysis|shell}" >&2
    exit 2
    ;;
esac
