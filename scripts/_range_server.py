"""Tiny Range-aware static server for local smoke testing.

Python 3.10's `http.server` ignores `Range` and returns the whole file with HTTP
200, which breaks the reviewer-finder's per-paper evidence fetch (and any other
range-based feature). GitHub Pages handles Range correctly, so this is only
needed locally.
"""
from __future__ import annotations

import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — stdlib name
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        m = _RANGE_RE.match(rng.strip())
        if not m:
            return super().do_GET()
        path = self.translate_path(self.path.split("?", 1)[0])
        if not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        if start >= size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            return
        end = min(end, size - 1)
        length = end - start + 1
        ctype = self.guess_type(path)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        # Permissive CORS so test fetches from other origins work too.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def end_headers(self):
        # Always advertise Range support, even on plain GET, so browsers know
        # they can issue partial requests.
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    directory = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    os.chdir(directory)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), RangeRequestHandler)
    print(f"[range-server] serving {directory} on http://localhost:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[range-server] shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
