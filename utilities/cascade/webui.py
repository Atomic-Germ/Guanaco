#!/usr/bin/env python3
"""Minimal escalation UI for the cascade.

Asks one or both models and shows the answers side by side, so the decision
"would the considered model have been better here?" can be made by looking
rather than by guessing. That comparison is the whole input the escalation
policy needs, and it is not something a throughput number can supply.

Deliberately not a patch of llama-server's WebUI: that page is generated and
embedded in the binary, so adding a model selector means regenerating it. A
150-line proxy that fronts both servers is easier to review and easier to change.

    python3 webui.py                  # then open http://127.0.0.1:8090
"""
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascade import MODELS, healthy, model_path, get  # noqa: E402

LISTEN_PORT = 8090

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Guanaco cascade</title>
<style>
 body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #14161a; color: #e6e6e6 }
 header { padding: 14px 20px; background: #1d2026; border-bottom: 1px solid #2b2f36 }
 h1 { font-size: 16px; margin: 0; font-weight: 600 }
 .sub { font-size: 12px; color: #8b929e; margin-top: 3px }
 .wrap { padding: 20px; max-width: 1500px; margin: 0 auto }
 textarea { width: 100%; min-height: 90px; background: #1d2026; color: #e6e6e6;
            border: 1px solid #2b2f36; border-radius: 6px; padding: 10px;
            font: 14px/1.5 ui-monospace, monospace; resize: vertical }
 button { background: #2d6cdf; color: white; border: 0; border-radius: 6px;
          padding: 8px 14px; margin: 10px 6px 0 0; cursor: pointer; font-size: 14px }
 button.ghost { background: #2b2f36 }
 button:disabled { opacity: .5; cursor: default }
 .cols { display: flex; gap: 16px; margin-top: 18px }
 .col { flex: 1; background: #1d2026; border: 1px solid #2b2f36; border-radius: 8px;
        padding: 14px; min-height: 220px; position: relative }
 .col h2 { font-size: 12px; margin: 0 0 10px; text-transform: uppercase;
           letter-spacing: .06em; color: #8b929e }
 .col.q h2 { color: #6ea8fe } .col.c h2 { color: #d98cff }
 .out { white-space: pre-wrap; font-size: 14px; word-break: break-word }
 .meta { position: absolute; top: 10px; right: 12px; font-size: 11px; color: #6b7280 }
 .empty { color: #4b5563; font-style: italic }
</style>
<div class="wrap">
 <header>
   <h1>Guanaco cascade</h1>
   <div class="sub">quick = fast and generic &middot; considered = slower and more
     capable one-shot. Not speculative decoding: this trades samples for
     capability, not for speed.</div>
 </header>
 <textarea id="q" placeholder="Ask something. Ctrl+Enter to send."></textarea>
 <div>
   <button onclick="go('quick')">Ask quick</button>
   <button onclick="go('considered')">Ask considered</button>
   <button class="ghost" onclick="go('both')">Ask both &mdash; compare</button>
   <button class="ghost" onclick="stopAll()">Stop considered</button>
 </div>
 <div class="cols">
   <div class="col q"><h2>quick</h2><div class="meta" id="qm"></div>
     <div class="out empty" id="qo">no answer yet</div></div>
   <div class="col c"><h2>considered</h2><div class="meta" id="cm"></div>
     <div class="out empty" id="co">no answer yet</div></div>
 </div>
</div>
<script>
const $ = id => document.getElementById(id);
$('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) go('both');
});

  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
function show(o, txt) {
  $(o).className = 'out';
  $(o).textContent = txt;
}
async function ask(role, prompt, n, out) {
  const r = await fetch('/ask?role=' + role + '&n=' + n, {
    method: 'POST', body: prompt });
  if (r.status === 202) return pollUntilReady(role, prompt, n, out);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
// On-demand: the model may not be running. /ask returns 202 and loads it in the
// background, so poll /status and retry. The timeout is deliberately long -
// a cold 111B load is minutes, not seconds - but a long wait that shows nothing
// is indistinguishable from a hang, so the elapsed time is displayed.
async function pollUntilReady(role, prompt, n, out) {
  const t0 = Date.now();
  for (let i = 0; i < 1800; i++) {          // 1800 x 2s = 60 min ceiling
    await new Promise(r => setTimeout(r, 2000));
    const s = await (await fetch('/status')).json();
    if (s[role] === 'up') {
      const r = await fetch('/ask?role=' + role + '&n=' + n, {
        method: 'POST', body: prompt });
      if (r.ok) return await r.json();
    }
    if (i % 5 === 0 && out) {
      const secs = Math.round((Date.now() - t0) / 1000);
      $(out).textContent = `loading the ${role} model... ${secs}s `
        + `(this is the on-demand cost: ~20s warm, 40-70s cold)`;
    }
  }
  throw new Error('gave up waiting for the model to load after 60 minutes');
}
async function go(which) {
  const prompt = $('q').value.trim();
  if (!prompt) return;
  const n = parseInt($('n') ? $('n').value : 160);
  const jobs = which === 'both' ? ['quick', 'considered'] : [which];
  for (const role of jobs) {
    const o = role === 'quick' ? 'qo' : 'co', m = role === 'quick' ? 'qm' : 'cm';
    $(o).className = 'out'; $(o).textContent = '...';
    ask(role, prompt, n, o).then(d => {
      show(o, d.content);
      $(m).textContent = d.rate.toFixed(2) + ' tok/s  ' + d.secs.toFixed(1) + 's';
    }).catch(e => { $(o).className = 'out empty';
                    $(o).textContent = 'failed: ' + e; });
  }
}
async function stopAll() {
  await fetch('/stop', { method: 'POST' });
  $('co').className = 'out empty'; $('co').textContent = 'unloaded';
  $('cm').textContent = '';
}
</script>
"""


class Handler(BaseHTTPRequestHandler):
    # The UI speaks in roles ("considered"); MODELS is keyed by concrete configs.
    # "considered" is an alias, resolved here, so --considered swaps which model
    # fills that role without touching the client.
    CONSIDERED = os.environ.get("CASCADE_CONSIDERED", "considered_dense")

    def resolve(self, role):
        if role == "considered":
            role = self.CONSIDERED
        return MODELS.get(role)

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, PAGE)
        if path == "/status":
            # Report the configured role, not every config that happens to share
            # a port - considered_dense and considered_moe both use 8082, so
            # listing them all says "up" for a model that was never loaded.
            spec = self.resolve("considered")
            out = {
                "quick": "up" if healthy(MODELS["quick"]["port"]) else "down",
                "considered": "up" if spec and healthy(spec["port"]) else "down",
                "considered_model": self.CONSIDERED,
            }
            return self._send(200, json.dumps(out), "application/json")
        self._send(404, "not found")

    def do_POST(self):
        role = "quick"
        n = 160
        if self.path.startswith("/ask"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            role = q.get("role", ["quick"])[0]
            n = int(q.get("n", [160])[0])
        elif self.path.startswith("/stop"):
            import cascade
            cascade.stop(self.CONSIDERED)
            return self._send(200, "ok", "text/plain")

        nbytes = int(self.headers.get("Content-Length", 0))
        prompt = self.rfile.read(nbytes).decode()
        spec = self.resolve(role)
        if spec is None:
            return self._send(400, f"unknown role: {role}")

        # On-demand means the model may simply not be running. Kick off the load
        # on a thread and tell the client to retry; the browser stays responsive
        # and shows progress rather than blocking for a minute.
        if not healthy(spec["port"]):
            self._spawn_load(role)
            return self._send(202, json.dumps({"status": "loading",
                                                "model": role}),
                              "application/json")

        try:
            body = json.dumps({"prompt": prompt, "n_predict": n,
                               "temperature": 0.2, "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{spec['port']}/completion", data=body,
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            d = json.load(urllib.request.urlopen(req, timeout=3600))
            out = {
                "content": d.get("content", ""),
                "rate": d.get("timings", {}).get("predicted_per_second", 0),
                "secs": time.time() - t0,
            }
            self._send(200, json.dumps(out), "application/json")
        except Exception as e:
            self._send(500, str(e), "text/plain")

    def _spawn_load(self, role):
        """Load the role in the background. It does NOT answer the prompt - the
        client retries once /status reports up, so the generation happens once."""
        import cascade
        start_role = self.CONSIDERED if role == "considered" else role
        spec = self.resolve(role)
        if spec is None:
            return

        def work():
            try:
                cascade.start(start_role)
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    print(f"cascade UI on http://127.0.0.1:{LISTEN_PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
