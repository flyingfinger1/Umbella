"""24: Browser-based review tool for the iNaturalist leaf images.

Starts a tiny local HTTP server, opens your browser to a single-image
review page, and lets you walk through each species with keyboard
shortcuts. Wrong/ambiguous images get moved to data/leaf_images/_trash/
(reversible) instead of being deleted outright.

Keyboard shortcuts:
    k  / Right Arrow / Space   — keep, next
    d  / Left  Arrow           — mark wrong (move to trash), next
    b  / Backspace             — back one image (undoes a previous trash)
    j                          — jump to next species
    Esc                        — show summary

Usage:
    .venv/Scripts/python.exe notebooks/24_review_leaf_images.py

Then open http://localhost:8765 in your browser. Close the terminal
window (Ctrl+C) when done.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data" / "leaf_images"
TRASH_ROOT = IMAGE_ROOT / "_trash"
PORT = 8765


def list_images() -> dict[str, list[str]]:
    """Per species, list of relative image paths (relative to IMAGE_ROOT)."""
    out: dict[str, list[str]] = {}
    if not IMAGE_ROOT.is_dir():
        return out
    for sp_dir in sorted(p for p in IMAGE_ROOT.iterdir() if p.is_dir() and p.name != "_trash"):
        files = sorted(f.name for f in sp_dir.glob("*.jpg"))
        if files:
            out[sp_dir.name] = files
    return out


HTML_PAGE = r"""<!doctype html>
<html lang="de"><head>
<meta charset="utf-8">
<title>Apiaceae review</title>
<style>
  body { background:#222; color:#eee; font-family:system-ui,sans-serif;
         margin:0; padding:0; display:flex; flex-direction:column;
         height:100vh; overflow:hidden; }
  header { padding:8px 16px; background:#111; display:flex;
           justify-content:space-between; align-items:center; }
  header .species { font-size:1.2em; font-weight:600; color:#7ec47e; }
  header .pos { color:#bbb; font-variant-numeric:tabular-nums; }
  main { flex:1; display:flex; align-items:center; justify-content:center;
         padding:8px; overflow:hidden; }
  main img { max-width:100%; max-height:100%; object-fit:contain;
             border:2px solid #444; }
  footer { padding:6px 16px; background:#111; font-size:0.9em; color:#aaa; }
  footer kbd { background:#333; border:1px solid #555; border-radius:3px;
               padding:1px 6px; font-family:monospace; }
  .stats { color:#7ec47e; }
  .deleted { border-color:#a44; box-shadow:0 0 16px #a44 inset; }
</style></head>
<body>
<header>
  <div class="species" id="species">loading…</div>
  <div class="pos" id="pos"></div>
  <div class="stats" id="stats"></div>
</header>
<main><img id="img" alt=""></main>
<footer>
  <kbd>K</kbd>/<kbd>→</kbd>/<kbd>Space</kbd> keep · <kbd>D</kbd>/<kbd>←</kbd> wrong (trash) ·
  <kbd>B</kbd>/<kbd>Backspace</kbd> back · <kbd>J</kbd> next species ·
  <kbd>Esc</kbd> summary
</footer>
<script>
let species = null;       // ordered list of species names
let imgs = {};            // { species: [filename, ...] }
let spIdx = 0;            // current species index
let imIdx = 0;            // index within current species
let history = [];         // [[spIdx, imIdx, action], ...] for "back"
let kept = 0, trashed = 0;

async function loadList() {
  const r = await fetch("/api/list");
  imgs = await r.json();
  species = Object.keys(imgs);
  if (species.length === 0) { alert("Keine Bilder in data/leaf_images/"); return; }
  spIdx = 0; imIdx = 0;
  show();
}

function show() {
  if (spIdx >= species.length) {
    document.getElementById("img").style.display = "none";
    document.getElementById("species").textContent = "Fertig";
    document.getElementById("pos").textContent = "";
    document.getElementById("stats").textContent = `kept ${kept}, trashed ${trashed}`;
    return;
  }
  const sp = species[spIdx];
  const list = imgs[sp];
  if (imIdx >= list.length) { spIdx++; imIdx = 0; show(); return; }
  document.getElementById("species").textContent = sp;
  document.getElementById("pos").textContent =
    `${imIdx + 1} / ${list.length}   (Spezies ${spIdx + 1}/${species.length})`;
  document.getElementById("stats").textContent =
    `kept ${kept}, trashed ${trashed}`;
  const img = document.getElementById("img");
  img.classList.remove("deleted");
  img.style.display = "";
  img.src = `/img/${sp}/${list[imIdx]}`;
  // preload next 3
  for (let k = 1; k <= 3; k++) {
    const n = list[imIdx + k];
    if (n) { const p = new Image(); p.src = `/img/${sp}/${n}`; }
  }
}

async function trashCurrent() {
  const sp = species[spIdx];
  const list = imgs[sp];
  if (imIdx >= list.length) return;
  const fname = list[imIdx];
  const r = await fetch(`/api/trash?species=${encodeURIComponent(sp)}&file=${encodeURIComponent(fname)}`,
                       { method: "POST" });
  if (!r.ok) { alert("Trash failed: " + await r.text()); return; }
  // remove from local list
  list.splice(imIdx, 1);
  history.push([spIdx, imIdx, "trash", sp, fname]);
  trashed++;
  show();
}

function keepCurrent() {
  const sp = species[spIdx];
  const list = imgs[sp];
  history.push([spIdx, imIdx, "keep", sp, list[imIdx]]);
  kept++;
  imIdx++;
  show();
}

async function back() {
  if (history.length === 0) return;
  const last = history.pop();
  const [si, ii, action, sp, fname] = last;
  if (action === "trash") {
    // restore from trash
    const r = await fetch(`/api/restore?species=${encodeURIComponent(sp)}&file=${encodeURIComponent(fname)}`,
                         { method: "POST" });
    if (!r.ok) { alert("Restore failed"); history.push(last); return; }
    imgs[sp].splice(ii, 0, fname);
    trashed--;
  } else {
    kept--;
  }
  spIdx = si; imIdx = ii;
  show();
}

function nextSpecies() {
  spIdx++; imIdx = 0; show();
}

function showSummary() {
  alert(`kept: ${kept}\ntrashed: ${trashed}\nremaining in trash:\n${history.filter(h => h[2]==='trash').map(h => h[3]+'/'+h[4]).join('\n')}`);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "k" || e.key === "ArrowRight" || e.key === " ")      keepCurrent();
  else if (e.key === "d" || e.key === "ArrowLeft")                    trashCurrent();
  else if (e.key === "b" || e.key === "Backspace")                    back();
  else if (e.key === "j")                                             nextSpecies();
  else if (e.key === "Escape")                                        showSummary();
  else return;
  e.preventDefault();
});

loadList();
</script></body></html>
"""


class ReviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # silence

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            return self._send_html()
        if u.path == "/api/list":
            return self._send_json(list_images())
        if u.path.startswith("/img/"):
            rel = u.path[len("/img/"):]
            return self._send_file(IMAGE_ROOT / rel)
        return self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        params = parse_qs(u.query)
        sp = params.get("species", [None])[0]
        f = params.get("file", [None])[0]
        if not sp or not f:
            return self.send_error(400, "missing species or file")
        if u.path == "/api/trash":
            src = IMAGE_ROOT / sp / f
            dst = TRASH_ROOT / sp / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                return self.send_error(404, "source not found")
            shutil.move(str(src), str(dst))
            return self._send_text("ok")
        if u.path == "/api/restore":
            src = TRASH_ROOT / sp / f
            dst = IMAGE_ROOT / sp / f
            if not src.exists():
                return self.send_error(404, "trash entry not found")
            shutil.move(str(src), str(dst))
            return self._send_text("ok")
        return self.send_error(404)

    def _send_html(self):
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if not IMAGE_ROOT.is_dir():
        sys.exit(f"no images in {IMAGE_ROOT}")
    counts = list_images()
    print(f"loaded species:")
    for sp, files in counts.items():
        print(f"  {sp}: {len(files)} images")
    print(f"\nopening http://localhost:{PORT} — Ctrl+C to stop")

    server = HTTPServer(("localhost", PORT), ReviewHandler)
    threading.Thread(target=lambda: webbrowser.open(f"http://localhost:{PORT}"),
                     daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. trashed images live in data/leaf_images/_trash/")


if __name__ == "__main__":
    main()
