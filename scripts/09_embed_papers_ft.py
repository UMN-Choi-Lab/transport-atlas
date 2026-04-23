#!/usr/bin/env python
"""Embed every paper with a fine-tuned SPECTER2 checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BATCH = int(os.environ.get("BATCH", "64"))
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = Path("/embed/specter2_ft/final")
DEFAULT_OUTPUT_PATH = Path("/embed/paper_embeddings_ft.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    in_path = repo / "data" / "interim" / "papers.parquet"
    out_path = args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved = {
        "input_path": str(in_path),
        "model_path": str(args.model_path),
        "output_path": str(out_path),
        "device": DEVICE,
        "batch_size": BATCH,
        "max_length": MAX_LEN,
    }
    if args.dry_run:
        print(resolved)
        return 0

    print(f"[embed-ft] model={args.model_path}  device={DEVICE}  batch={BATCH}", flush=True)
    print(f"[embed-ft] reading {in_path}", flush=True)
    papers = pd.read_parquet(in_path)
    print(f"[embed-ft] papers: {len(papers):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(DEVICE).eval()
    model = model.half() if DEVICE == "cuda" else model
    sep = tokenizer.sep_token or "[SEP]"

    def _str(v):
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v).strip()

    def _mk_text(r):
        t = _str(r.get("title"))
        a = _str(r.get("abstract"))
        return f"{t}{sep}{a}" if a else t

    embs = np.empty((len(papers), 768), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(papers), BATCH), desc="embed"):
            batch = papers.iloc[i : i + BATCH]
            texts = [_mk_text(r) for _, r in batch.iterrows()]
            tok = tokenizer(
                texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
            ).to(DEVICE)
            out = model(**tok)
            cls = out.last_hidden_state[:, 0, :]
            embs[i : i + len(batch)] = cls.cpu().float().numpy().astype(np.float16)
    elapsed = time.time() - t0
    print(f"[embed-ft] done in {elapsed/60:.1f} min  ({len(papers)/elapsed:.1f} papers/s)", flush=True)

    df = pd.DataFrame({
        "paper_id": papers["paper_id"].tolist(),
        "emb": embs.astype(np.float32).tolist(),
    })
    df.to_parquet(out_path, index=False)
    print(f"[embed-ft] wrote {out_path} ({len(df):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
