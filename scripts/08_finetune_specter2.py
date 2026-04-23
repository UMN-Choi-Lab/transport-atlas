#!/usr/bin/env python
"""Continued MLM pretraining for SPECTER2 on the transportation corpus.

Uses a plain PyTorch training loop (no HF Trainer / accelerate) so the
existing transport-atlas-embed:v1 image works without a rebuild.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "allenai/specter2_base"
DEFAULT_OUTPUT_DIR = Path("/embed/specter2_ft")
DEFAULT_INPUT_PATH = Path("data/interim/papers.parquet")
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 3
DEFAULT_YEAR_MAX = 2019
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX)
    p.add_argument("--include-2020-2025", action="store_true")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _clean(v: object) -> str:
    if not isinstance(v, str):
        return ""
    return v.strip()


def build_text(title: object, abstract: object) -> str:
    t = _clean(title)
    a = _clean(abstract)
    return f"{t} [SEP] {a}" if a else t


def load_texts(in_path: Path, year_max: int) -> list[str]:
    papers = pd.read_parquet(in_path)
    yr = pd.to_numeric(papers["year"], errors="coerce")
    papers = papers[yr.fillna(-math.inf) <= year_max]
    texts = [build_text(r.get("title"), r.get("abstract"))
             for _, r in papers.iterrows()]
    return [t for t in texts if t]


class TokenizedDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int) -> None:
        enc = tokenizer(texts, truncation=True, max_length=max_len,
                        return_attention_mask=True)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def resolve_cfg(args: argparse.Namespace, repo: Path) -> dict:
    effective_year_max = 2025 if args.include_2020_2025 else args.year_max
    ip = args.input_path if args.input_path.is_absolute() else repo / args.input_path
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "input_path": str(ip),
        "output_dir": str(args.output_dir),
        "year_max": int(effective_year_max),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "max_seq_length": int(args.max_seq_length),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "fp16": bool(torch.cuda.is_available()),
    }


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg = resolve_cfg(args, repo)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    torch.manual_seed(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    year_max = cfg["year_max"]
    print(f"[finetune] model={MODEL_NAME} device={DEVICE} year_max={year_max}", flush=True)
    print(f"[finetune] reading {cfg['input_path']}", flush=True)
    texts = load_texts(Path(cfg["input_path"]), year_max)
    print(f"[finetune] usable_docs={len(texts):,}", flush=True)
    if not texts:
        raise SystemExit("No documents.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train()

    ds = TokenizedDataset(texts, tokenizer, cfg["max_seq_length"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collator, pin_memory=True,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(0.1 * total_steps))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"])

    log_rows: list[dict] = []
    saved_epochs: list[Path] = []
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        run_loss = 0.0
        run_n = 0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["fp16"]):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += float(loss.item()); run_n += 1
            if step % 50 == 0:
                gstep = (epoch - 1) * steps_per_epoch + step
                log_rows.append({"step": gstep, "loss": float(loss.item()),
                                 "epoch": epoch + step / max(1, steps_per_epoch)})
                pbar.set_postfix(loss=f"{run_loss / run_n:.4f}")

        epoch_dir = run_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        saved_epochs.append(epoch_dir)
        print(f"[finetune] saved {epoch_dir} (mean_loss={run_loss / max(1, run_n):.4f})", flush=True)

    wall = time.time() - t0
    final_link = output_dir / "final"
    if final_link.is_symlink() or final_link.exists():
        try:
            final_link.unlink()
        except IsADirectoryError:
            import shutil; shutil.rmtree(final_link)
    os.symlink(saved_epochs[-1].resolve(), final_link)

    (run_dir / "training_log.json").write_text(json.dumps({
        "model_name": MODEL_NAME,
        "input_path": cfg["input_path"],
        "run_dir": str(run_dir),
        "final_checkpoint": str(saved_epochs[-1]),
        "final_symlink": str(final_link),
        "num_docs": len(ds),
        "num_epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": warmup_steps,
        "max_seq_length": cfg["max_seq_length"],
        "year_max": year_max,
        "wall_time_sec": wall,
        "loss_curve": log_rows,
    }, indent=2))
    print(f"[finetune] wall_clock={wall/60:.1f}min final->{saved_epochs[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
