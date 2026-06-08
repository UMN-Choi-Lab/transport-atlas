#!/usr/bin/env python
"""Web UI for manually nudging label positions in Figures 2 (coauthor
overview + LCC labeled) and 4 (topic space).

Flow:
  1. On first run, subprocesses `06_render_atlas_views.py` to bake
     background SVGs + label specs into `.label_editor_cache/`.
  2. Starts a localhost HTTP server.
  3. User drags labels in the browser, clicks Save → writes YAML back
     to `paper/analysis/label_overrides.yaml`.
  4. User re-runs `06_render_atlas_views.py` to regenerate the PDFs
     with manual positions pinned.

Usage:
    python paper/analysis/edit_labels.py                # serve on 127.0.0.1:8766
    python paper/analysis/edit_labels.py --port 9000
    python paper/analysis/edit_labels.py --host 0.0.0.0 # remote access (be cautious)
    python paper/analysis/edit_labels.py --rebake       # force regenerate caches

Override file schema (label_overrides.yaml):
    coauthor_overview:
      "15": [x, y]     # xytext for community C15 in data coords
    lcc_labeled:
      "33329": [x, y]  # xytext for node 33329 (Wu, Yuankai)
    topic_space:
      "5":  [x, y]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "paper" / "analysis"
CACHE_DIR = ANALYSIS / ".label_editor_cache"
OVERRIDES_PATH = ANALYSIS / "label_overrides.yaml"
TEMPLATE_PATH = ANALYSIS / "label_editor_index.html"

FIGURES = {
    "coauthor_overview": "Fig 2(a) — coauthor overview (C0–C11)",
    "lcc_labeled":       "Fig 2(b) — LCC author names",
    "topic_space":       "Fig 4 — topic space (S0–S11)",
}


def ensure_cache(rebake: bool = False) -> None:
    need = rebake or not all(
        (CACHE_DIR / f"{k}.bg.svg").exists()
        and (CACHE_DIR / f"{k}.specs.json").exists()
        for k in FIGURES
    )
    if not need:
        return
    print("[edit-labels] baking caches via 06_render_atlas_views.py ...")
    ret = subprocess.call(
        [sys.executable, str(ANALYSIS / "06_render_atlas_views.py")],
        cwd=ROOT,
    )
    if ret != 0:
        print(f"[edit-labels] renderer exited with status {ret}", file=sys.stderr)
        sys.exit(ret)


def load_overrides() -> dict:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(OVERRIDES_PATH.read_text()) or {}
    except Exception as e:
        print(f"[edit-labels] WARN reading overrides: {e}")
        return {}


_YAML_HEADER = (
    "# Manual label position overrides for Figures 2 and 4.\n"
    "# Generated / edited by paper/analysis/edit_labels.py — hand-editing OK.\n"
    "# Each entry: label_id -> [x, y] in the figure's data coords.\n"
    "# Delete an entry (or the whole figure block) to restore auto-layout.\n"
    "\n"
)


def write_overrides(doc: dict) -> None:
    """Emit YAML with a stable, human-readable order. One entry per line."""
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [_YAML_HEADER.rstrip() + "\n"]
    for fkey in ("coauthor_overview", "lcc_labeled", "topic_space"):
        sub = doc.get(fkey) or {}
        if not sub:
            continue
        lines.append(f"{fkey}:\n")
        for lid in sorted(sub.keys(), key=lambda s: (len(s), s)):
            xy = sub[lid]
            lines.append(f'  "{lid}": [{float(xy[0]):.4f}, {float(xy[1]):.4f}]\n')
        lines.append("\n")
    OVERRIDES_PATH.write_text("".join(lines))


def handle_save(payload: dict) -> dict:
    """payload = {figure_key: {label_id: [x, y], ...}, ...}"""
    doc = load_overrides()
    for fkey, subs in payload.items():
        if fkey not in FIGURES:
            continue
        cur = doc.get(fkey) or {}
        # None value → remove an override (user reset that label to auto).
        for lid, xy in subs.items():
            if xy is None:
                cur.pop(str(lid), None)
            else:
                cur[str(lid)] = [float(xy[0]), float(xy[1])]
        if cur:
            doc[fkey] = cur
        else:
            doc.pop(fkey, None)
    write_overrides(doc)
    return {"ok": True, "path": str(OVERRIDES_PATH),
            "n_pinned": {k: len(v) for k, v in doc.items()}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        return

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):  # noqa: N802
        p = self.path.split("?", 1)[0]
        if p == "/" or p == "/index.html":
            try:
                body = TEMPLATE_PATH.read_bytes()
            except FileNotFoundError:
                self._send(500, b"template missing", "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if p == "/api/figures":
            self._json({
                "figures": [{"key": k, "title": t} for k, t in FIGURES.items()],
                "overrides": load_overrides(),
            })
            return
        if p.startswith("/api/figure/"):
            key = p.removeprefix("/api/figure/").removesuffix(".json")
            fp = CACHE_DIR / f"{key}.specs.json"
            if not fp.exists():
                self._send(404, b"not baked", "text/plain")
                return
            self._send(200, fp.read_bytes(), "application/json")
            return
        if p.startswith("/bg/"):
            key = p.removeprefix("/bg/").removesuffix(".svg")
            fp = CACHE_DIR / f"{key}.bg.svg"
            if not fp.exists():
                self._send(404, b"not baked", "text/plain")
                return
            self._send(200, fp.read_bytes(), "image/svg+xml")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._json({"ok": False, "error": f"bad JSON: {e}"}, 400)
            return
        try:
            out = handle_save(payload)
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)
            return
        self._json(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--rebake", action="store_true",
                    help="Force regenerate background SVGs + specs.")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ensure_cache(rebake=args.rebake)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/"
    print(f"[edit-labels] serving at {url}")
    print(f"[edit-labels] overrides file: {OVERRIDES_PATH}")
    print("[edit-labels] Ctrl-C to stop.")
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[edit-labels] bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
