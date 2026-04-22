#!/usr/bin/env python
"""Quick SPECTER2 sanity check — embed 4 papers, confirm the output dim + CUDA path.

Use before running the full 05_embed_papers.py to catch model-loading issues.
"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "allenai/specter2_base"
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)

print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModel.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(device).half().eval() if device == "cuda" else m.to(device).eval()
sep = tok.sep_token or "[SEP]"

papers = [
    ("TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning",
     "We propose a trajectory generation framework based on GAIL..."),
    ("Probabilistic Traffic Forecasting with Dynamic Regression",
     "A scalable approach to probabilistic forecasting of traffic states..."),
    ("Shared Autonomous Vehicle Planning",
     "Fleet-size and parking-infrastructure optimization for SAVs..."),
    ("Cooking Pasta With Emmental Cheese",
     "Recipe analysis of pasta-cheese interactions..."),
]
texts = [t + sep + a for t, a in papers]
tok_in = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    out = m(**tok_in)
E = out.last_hidden_state[:, 0, :].float().cpu().numpy()
print(f"embeddings shape: {E.shape}  dtype={E.dtype}", flush=True)

# Cosine similarities — transportation papers (0-2) should cluster together,
# the pasta paper (3) should be further away from all of them.
import numpy as np
E = E / np.linalg.norm(E, axis=1, keepdims=True)
sims = E @ E.T
print("pairwise cosine:", flush=True)
labels = ["TrajGAIL", "ProbForecast", "SAV", "Pasta"]
for i, a in enumerate(labels):
    row = "  " + f"{a:>13}  " + "  ".join(f"{sims[i,j]:.3f}" for j in range(len(labels)))
    print(row, flush=True)
avg_trans = (sims[0,1] + sims[0,2] + sims[1,2]) / 3
avg_pasta = (sims[0,3] + sims[1,3] + sims[2,3]) / 3
print(f"\naverage intra-transport sim: {avg_trans:.3f}", flush=True)
print(f"average transport-to-pasta sim: {avg_pasta:.3f}", flush=True)
assert avg_trans > avg_pasta, f"transportation papers should cluster tighter than pasta ({avg_trans} vs {avg_pasta})"
print("OK — SPECTER2 semantic clustering looks right.", flush=True)
