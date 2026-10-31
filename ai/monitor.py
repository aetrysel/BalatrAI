#!/usr/bin/env python3
"""
Instance monitor: a local web page showing every Balatro instance's latest screen + status

Reads instances/<name>/screen.png and status.json (written by RLBridge/headless.lua).
Optionally records each instance's snapshots into a timelapse video.

Usage:
    python -m ai.monitor                 # http://localhost:8765
    python -m ai.monitor --record        # also write recordings/<name>_<time>.mp4 (one frame per snapshot)
"""
import argparse
import json
import subprocess
import threading
import time
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .utils.instances import INSTANCES_ROOT

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Balatro Instances</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#14161a; --card:#1f232a; --text:#e6e6e6; --muted:#8a919c; --ok:#4caf7d; --warn:#e0a83a; --bad:#e05a4f; }
  body { margin:0; padding:16px; background:var(--bg); color:var(--text); font:14px system-ui, sans-serif; }
  h1 { font-size:18px; margin:0 0 12px; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(360px, 1fr)); gap:12px; }
  .card { background:var(--card); border-radius:8px; overflow:hidden; border:2px solid transparent; }
  .card.stale { border-color:var(--bad); }
  .card img { width:100%; aspect-ratio:16/9; object-fit:contain; background:#000; display:block; }
  .meta { padding:8px 10px; display:flex; flex-wrap:wrap; gap:4px 14px; }
  .name { font-weight:600; width:100%; display:flex; justify-content:space-between; }
  .dot { color:var(--ok); } .stale .dot { color:var(--bad); }
  .k { color:var(--muted); }
</style></head><body>
<h1>Balatro instances <span class="k" id="summary"></span></h1>
<div id="grid"></div>
<script>
const STATES = {1:'Selecting hand',2:'Hand played',3:'Drawing',4:'Game over',5:'Shop',6:'Play tarot',7:'Blind select',
  8:'Round eval',9:'Tarot pack',10:'Planet pack',11:'Menu',13:'Splash',15:'Spectral pack',17:'Standard pack',
  18:'Buffoon pack',19:'New round',999:'Booster pack'};
const cards = {};
async function refresh() {
  const names = await (await fetch('/api/instances')).json();
  const grid = document.getElementById('grid');
  let live = 0;
  for (const n of names) {
    if (!cards[n]) {
      const el = document.createElement('div');
      el.className = 'card';
      el.innerHTML = `<img alt="${n} screen"><div class="meta"></div>`;
      grid.appendChild(el); cards[n] = el;
    }
    const el = cards[n];
    let s = {};
    try { s = await (await fetch(`/files/${n}/status.json?t=${Date.now()}`)).json(); } catch (e) {}
    const age = s.time ? Math.round(Date.now()/1000 - s.time) : null;
    const stale = age === null || age > 15;
    if (!stale) live++;
    el.classList.toggle('stale', stale);
    el.querySelector('img').src = `/files/${n}/screen.png?t=${Date.now()}`;
    el.querySelector('.meta').innerHTML =
      `<div class="name"><span>${n}${s.headless === false ? ' (window)' : ''}</span><span class="dot">● ${age === null ? 'no data' : age + 's ago'}</span></div>` +
      `<span><span class="k">state</span> ${STATES[s.state] || s.state}</span>` +
      `<span><span class="k">ante</span> ${s.ante ?? '-'}</span><span><span class="k">round</span> ${s.round ?? '-'}</span>` +
      `<span><span class="k">$</span>${s.dollars ?? '-'}</span><span><span class="k">jokers</span> ${s.jokers ?? '-'}</span>` +
      `<span><span class="k">blind</span> ${s.blind || '-'}</span><span><span class="k">fps</span> ${s.fps ?? '-'}</span>`;
  }
  document.getElementById('summary').textContent = `${live}/${names.length} reporting`;
}
refresh(); setInterval(refresh, 2000);
</script></body></html>
"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html")
        if self.path == "/api/instances":
            names = sorted(p.name for p in INSTANCES_ROOT.iterdir() if (p / "status.json").exists()) \
                if INSTANCES_ROOT.exists() else []
            return self._send(json.dumps(names).encode(), "application/json")
        if self.path.startswith("/files/"):
            self.path = self.path[len("/files"):]
            return super().do_GET()
        self.send_error(404)

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def record_instance(name: str, interval: float, out_dir: Path, stop: threading.Event):
    """Pipe each new snapshot of one instance into ffmpeg as one video frame."""
    screen = INSTANCES_ROOT / name / "screen.png"
    out = out_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", "10", "-i", "-",
         "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    last_mtime = 0.0
    while not stop.is_set():
        try:
            mtime = screen.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                proc.stdin.write(screen.read_bytes())
        except (FileNotFoundError, BrokenPipeError):
            pass
        stop.wait(interval / 2)
    proc.stdin.close()
    proc.wait()
    print(f"🎞  {out}")


def main():
    p = argparse.ArgumentParser(description="Monitor Balatro instances")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--record", action="store_true", help="record each instance's snapshots to recordings/*.mp4")
    p.add_argument("--snapshot-secs", type=float, default=2.0, help="snapshot interval the instances use")
    args = p.parse_args()

    stop = threading.Event()
    recorders = {}
    if args.record:
        out_dir = Path("recordings")
        out_dir.mkdir(exist_ok=True)

        def spawn_recorders():
            while not stop.is_set():
                for inst in INSTANCES_ROOT.glob("*/screen.png"):
                    name = inst.parent.name
                    if name not in recorders:
                        t = threading.Thread(target=record_instance, args=(name, args.snapshot_secs, out_dir, stop))
                        t.start()
                        recorders[name] = t
                stop.wait(5)
        threading.Thread(target=spawn_recorders, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(INSTANCES_ROOT.resolve())))
    print(f"📺 Monitor at http://localhost:{args.port}{'  (recording → recordings/)' if args.record else ''}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in recorders.values():
            t.join()
        server.server_close()


if __name__ == "__main__":
    main()
