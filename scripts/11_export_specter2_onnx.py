#!/usr/bin/env python
"""Export allenai/specter2_base to int8 ONNX for browser use, push to HF Hub.

Why: transformers.js (browser) needs the model in ONNX form, served from a HF
Hub repo (or local path). No-one has published an ONNX of specter2_base yet —
we make one and push it under the user's account so the static atlas can
load it via @xenova/transformers.

Steps:
    1. Read HF_TOKEN from project .env (config.hf_token()).
    2. Determine username via huggingface_hub.HfApi().whoami().
    3. Export float32 ONNX with optimum.
    4. Quantize int8 (avx512_vnni) for ~3-4x size reduction (~110 MB).
    5. Lay files out in the transformers.js convention (root metadata + ./onnx/).
    6. Create the target repo (idempotent) and push the folder.

Output:
    Local mirror at /embed/specter2_onnx_export/ (cached, idempotent).
    Remote repo: <username>/<HF_REPO_NAME or default>.

Run inside the project Docker image:
    ./docker/run_embed.sh export-specter2-onnx
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_atlas.utils import config

MODEL_ID = "allenai/specter2_base"
DEFAULT_REPO_NAME = os.environ.get("HF_REPO_NAME", "specter2-base-onnx-web")
EMBED_DIR = Path(os.environ.get("EMBED_OUT", "/data2/chois/transport-atlas"))
EXPORT_DIR = EMBED_DIR / "specter2_onnx_export"
QUANT_DIR = EMBED_DIR / "specter2_onnx_quantized"
PUSH_DIR = EMBED_DIR / "specter2_onnx_push"


def _ensure_export() -> None:
    """Run the float32 ONNX export if the cached files aren't there.

    Use ``main_export`` (graph-only export) rather than
    ``ORTModelForFeatureExtraction.from_pretrained(..., export=True)`` because the
    latter spins up an ONNXRuntime *session* on the freshly-exported model, and
    onnxruntime 1.25 references ``torch.int4`` (added in torch 2.6) during session
    init — which crashes on torch 2.5 (what's pinned in the embed image).
    """
    onnx_path = EXPORT_DIR / "model.onnx"
    if onnx_path.exists():
        print(f"[onnx] cached export found at {EXPORT_DIR}; skipping export step",
              flush=True)
        return
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    from optimum.exporters.onnx import main_export
    print(f"[onnx] exporting {MODEL_ID} → ONNX (graph-only; ~1-2 min)",
          flush=True)
    main_export(
        model_name_or_path=MODEL_ID,
        output=str(EXPORT_DIR),
        task="feature-extraction",
        opset=14,
    )
    if not onnx_path.exists():
        raise FileNotFoundError(f"main_export produced no model.onnx in {EXPORT_DIR}; "
                                f"contents: {sorted(p.name for p in EXPORT_DIR.iterdir())}")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[onnx] wrote {onnx_path}  ({size_mb:.1f} MB float32)", flush=True)


def _ensure_quantized() -> None:
    quant_path = QUANT_DIR / "model_quantized.onnx"
    if quant_path.exists():
        print(f"[onnx] cached quantized model at {QUANT_DIR}; skipping quantize step",
              flush=True)
        return
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    print("[onnx] dynamic-quantizing to int8 (avx512_vnni) ...", flush=True)
    quantizer = ORTQuantizer.from_pretrained(EXPORT_DIR, file_name="model.onnx")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=QUANT_DIR, quantization_config=qconfig)
    # optimum saves as `model_quantized.onnx` in QUANT_DIR.
    if not quant_path.exists():
        # Some optimum versions save with a different stem.
        cands = list(QUANT_DIR.glob("*.onnx"))
        if not cands:
            raise FileNotFoundError(f"quantize produced no .onnx in {QUANT_DIR}")
        cands[0].rename(quant_path)
    size_mb = quant_path.stat().st_size / (1024 * 1024)
    print(f"[onnx] wrote {quant_path}  ({size_mb:.1f} MB int8)", flush=True)


def _stage_push_dir() -> None:
    """Lay files out the way transformers.js expects:
        repo/
          config.json, tokenizer*.json, special_tokens_map.json, vocab.txt
          onnx/model.onnx           (float32)
          onnx/model_quantized.onnx (int8)
    """
    if PUSH_DIR.exists():
        shutil.rmtree(PUSH_DIR)
    (PUSH_DIR / "onnx").mkdir(parents=True, exist_ok=True)
    # Metadata files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.txt"]:
        src = EXPORT_DIR / fname
        if src.exists():
            shutil.copy2(src, PUSH_DIR / fname)
    # ONNX files
    shutil.copy2(EXPORT_DIR / "model.onnx", PUSH_DIR / "onnx" / "model.onnx")
    shutil.copy2(QUANT_DIR / "model_quantized.onnx",
                 PUSH_DIR / "onnx" / "model_quantized.onnx")
    # Drop a README so the repo isn't bare
    readme = PUSH_DIR / "README.md"
    readme.write_text(
        "# specter2-base for transformers.js\n\n"
        f"ONNX export of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) for use\n"
        "in the browser via [@xenova/transformers](https://github.com/xenova/transformers.js).\n\n"
        "- `onnx/model.onnx` — float32\n"
        "- `onnx/model_quantized.onnx` — int8 dynamic-quantized\n\n"
        "Pooling: `[CLS]` (first token of last hidden state).\n"
        "Input: `<title>[SEP]<abstract>`, max length 512 tokens.\n\n"
        "License: same as the upstream allenai/specter2_base (Apache-2.0).\n"
    )
    print(f"[onnx] staged push dir at {PUSH_DIR}", flush=True)
    for p in sorted(PUSH_DIR.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(PUSH_DIR)}  ({sz:.1f} MB)", flush=True)


def _push_to_hub() -> str:
    token = config.hf_token()
    if not token:
        print("[onnx] HF_TOKEN missing — set it in <repo>/.env, then re-run.",
              file=sys.stderr)
        sys.exit(2)
    from huggingface_hub import HfApi, create_repo, upload_folder
    api = HfApi(token=token)
    me = api.whoami()
    user = me.get("name") or me.get("email") or ""
    if not user:
        print(f"[onnx] could not determine HF username from token; whoami={me}",
              file=sys.stderr)
        sys.exit(2)
    repo_id = f"{user}/{DEFAULT_REPO_NAME}"
    print(f"[onnx] target repo: {repo_id}", flush=True)
    create_repo(repo_id, token=token, exist_ok=True, private=False)
    print(f"[onnx] uploading {PUSH_DIR} → {repo_id} ...", flush=True)
    upload_folder(
        folder_path=str(PUSH_DIR),
        repo_id=repo_id,
        token=token,
        commit_message="Export allenai/specter2_base to ONNX for transformers.js",
    )
    print(f"[onnx] done. Browser: pipeline('feature-extraction', '{repo_id}', "
          "{ quantized: true })", flush=True)
    return repo_id


def main() -> int:
    _ensure_export()
    _ensure_quantized()
    _stage_push_dir()
    repo_id = _push_to_hub()
    # Persist the resolved repo id so the render step can inject it.
    out = EMBED_DIR / "specter2_repo_id.txt"
    out.write_text(repo_id + "\n")
    print(f"[onnx] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
