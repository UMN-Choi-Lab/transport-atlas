#!/usr/bin/env bash
# Transport Atlas — scheduled incremental update (+ optional deploy).
#
# This is the entry point for the cron job. It runs every stage inside the
# transport-atlas-embed Docker image (via docker/run_embed.sh) so the host
# login shell never runs the pipeline directly, then optionally syncs the built
# site to the gh-pages repo and pushes.
#
# Cadence + lookback live in config/pipeline.yaml (update.cadence_cron /
# update.lookback_days). The front-page "Updated <cadence>" note reads the same
# config, so changing the schedule there keeps the site claim honest.
#
# Install (weekly, Mondays 06:00 America/Chicago = 11:00 UTC):
#   crontab -e
#   0 11 * * 1 /home/chois/gitsrcs/transportation/scripts/scheduled_update.sh --deploy \
#       >> /home/chois/gitsrcs/transportation/data/processed/_cron.log 2>&1
#
# Manual catch-up (no deploy — inspect site/ first):
#   ./scripts/scheduled_update.sh
# Manual catch-up + deploy:
#   ./scripts/scheduled_update.sh --deploy
# Override the publication-date window for a one-off wide sweep:
#   LOOKBACK_DAYS=180 ./scripts/scheduled_update.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${REPO_ROOT}/docker/run_embed.sh"
GHPAGES="/home/chois/gitsrcs/choi-seongjin.github.io"
DEST="${GHPAGES}/transport-atlas/"

DEPLOY=0
# GPU: honour an explicit GPU=N, else auto-pick the device with the most free
# memory so the recurring job never collides with whoever is on GPU 0.
pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2 -n | head -1 | tr -d ' ' | cut -d, -f1
}
if [ -z "${GPU:-}" ]; then GPU="$(pick_free_gpu)"; fi
export GPU="${GPU:-0}"                        # heavy stages (embed/similarity) want a GPU
# Ingest mode: "full" (default, reliable) re-pulls every source with no date
# filter, so it catches preprint-promoted / placeholder-dated papers that the
# publication-date "incremental" mode silently misses. See docker/run_embed.sh.
MODE="${INGEST_MODE:-full}"
LOOKBACK="${LOOKBACK_DAYS:-90}"              # only used when MODE=incremental
for a in "$@"; do
  case "$a" in
    --deploy) DEPLOY=1 ;;
    --incremental) MODE="incremental" ;;
    --full) MODE="full" ;;
    --lookback=*) LOOKBACK="${a#*=}" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

ts()   { date "+%Y-%m-%d %H:%M:%S"; }
step() { echo; echo "=== [$(ts)] $* ==="; }

cd "$REPO_ROOT"

if [ "$MODE" = "incremental" ]; then
  step "1/7 OpenAlex incremental top-up (publication window ${LOOKBACK}d — fast, may miss placeholder-dated papers)"
  "$RUN" ingest-recent --lookback-days "$LOOKBACK"
else
  step "1/7 OpenAlex full re-pull (reliable — catches re-associated / preprint-promoted papers)"
  "$RUN" ingest-full
fi

step "2/7 corpus rebuild (dedupe -> aggregate -> coauthor graph -> annotate)"
"$RUN" corpus

step "3/7 embed new papers (SPECTER2, checkpointed — only new paper_ids)"
"$RUN" embed

step "4/7 author similarity / topic coords / communities / trajectories"
"$RUN" similarity

step "5/7 reflag phantom neighbours"
"$RUN" reflag-phantoms

step "6/7 reviewer + paper index"
"$RUN" reviewer-index

step "7/7 render static site"
"$RUN" render

if [ "$DEPLOY" -eq 1 ]; then
  step "deploy -> gh-pages"
  rsync -av --delete "${REPO_ROOT}/site/" "$DEST"
  cd "$GHPAGES"
  git add transport-atlas/
  if git diff --cached --quiet; then
    echo "no site changes to commit"
  else
    git commit -m "transport-atlas: scheduled update $(date +%Y-%m-%d)"
    git push origin gh-pages
  fi
else
  step "skip deploy (pass --deploy to sync + push to gh-pages)"
fi

step "done"
