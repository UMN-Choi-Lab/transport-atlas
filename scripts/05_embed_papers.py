#!/usr/bin/env python
"""Embed every paper with SPECTER2 base encoder.

Input:  data/interim/papers.parquet  (dedup'd corpus, post-frontmatter-filter)
Output: /data2/chois/transport-atlas/paper_embeddings.parquet  (paper_id, emb)

SPECTER2 base is a scibert-style encoder trained on paper-paper citation pairs,
so the [CLS] embedding of `<title>[SEP]<abstract>` clusters semantically related
papers much tighter than generic sentence transformers.

Runs ~50ms/paper on an RTX 6000 Ada w/ batch 64 -> ~60 min for 75k papers.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL = "allenai/specter2_base"
OUT_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
BATCH = int(os.environ.get("BATCH", "64"))
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    in_path = repo / "data" / "interim" / "papers.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "paper_embeddings.parquet"

    print(f"[embed] model={MODEL}  device={DEVICE}  batch={BATCH}", flush=True)
    print(f"[embed] reading {in_path}", flush=True)
    papers = pd.read_parquet(in_path)
    print(f"[embed] papers: {len(papers):,}", flush=True)

    # Checkpoint: skip IDs already embedded
    done: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["paper_id"].tolist())
        print(f"[embed] resume: {len(done):,} already embedded", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()
    model = model.half() if DEVICE == "cuda" else model  # fp16 on GPU
    sep = tokenizer.sep_token or "[SEP]"

    # Prepare text per paper
    def _str(v):
        # Parquet NaN comes through as float('nan') — coerce before .strip()
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v).strip()

    def _mk_text(r):
        t = _str(r.get("title"))
        a = _str(r.get("abstract"))
        return f"{t}{sep}{a}" if a else t

    papers = papers[~papers["paper_id"].isin(done)].reset_index(drop=True)
    print(f"[embed] to embed this run: {len(papers):,}", flush=True)
    if len(papers) == 0:
        print("[embed] nothing to do", flush=True)
        return 0

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
            # [CLS] token embedding
            cls = out.last_hidden_state[:, 0, :]
            embs[i : i + len(batch)] = cls.cpu().float().numpy().astype(np.float16)
    elapsed = time.time() - t0
    print(f"[embed] done in {elapsed/60:.1f} min  ({len(papers)/elapsed:.1f} papers/s)", flush=True)

    df = pd.DataFrame({
        "paper_id": papers["paper_id"].tolist(),
        # Store as list[float16] — pyarrow will use fixed-size list
        "emb": [emb for emb in embs],
    })
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(f"[embed] wrote {out_path} ({len(df):,} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
