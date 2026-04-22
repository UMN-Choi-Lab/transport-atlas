# Transport Atlas

Static paper-atlas for transportation research venues (TR A-F, IEEE T-ITS / T-IV / ITS Mag / IV Symposium).

Inspired by [robopaper-atlas](https://gisbi-kim.github.io/robopaper-atlas/).

## Install

```bash
cd transportation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# One-shot, resumable
python scripts/refresh.py

# Or per phase
python scripts/01_ingest.py --source openalex
python scripts/01_ingest.py --source ieee
python scripts/01_ingest.py --source elsevier
python scripts/02_dedupe.py
python scripts/03_graph.py
python scripts/04_render.py

# Preview
cd site && python -m http.server 8000
```

## Data layout

- `data/raw/{openalex,ieee,elsevier}/<venue>.jsonl` — per-source ingest, checkpointed
- `data/interim/papers.parquet`, `authors.parquet` — dedup'd
- `data/processed/*.json` — site-ready

## API keys

Read from `~/.claude/mcp-servers/refcheck/.env`. No keys in this repo.
