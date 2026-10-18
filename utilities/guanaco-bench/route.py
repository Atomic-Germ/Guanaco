#!/usr/bin/env python3
"""Cascade router: send a prompt to one of two models.

Deliberately has no escalation logic. The plumbing and the policy are separate
concerns, and the only thing between "this works" and "this is useful" is which
model gets which request - which needs a labelled workload, not a heuristic
invented here.

What it does know is the measured cost structure:

  small ~6.4 tok/s   large ~1.3 tok/s   (4.8x apart)

so at f = 0.8 escalation the blend is ~3.6 tok/s, about 2.7x pure-large. Whether
that is worth it depends entirely on the escalation rate, which is the user's
call and the one number this tool cannot supply.

usage:
  route.py -t small|large "prompt"
  route.py -t small -f file.txt
  route.py -t large -n 200 "longer answer please"
"""
import argparse
import json
import sys
import time
import urllib.request

PORTS = {"small": 8081, "large": 8082}


def complete(port, prompt, n_predict, temperature):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target", choices=list(PORTS), required=True)
    ap.add_argument("-n", "--n-predict", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("-f", "--file")
    ap.add_argument("prompt", nargs="?")
    a = ap.parse_args()

    if a.file:
        prompt = open(a.file).read()
    elif a.prompt:
        prompt = a.prompt
    else:
        ap.error("need a prompt or -f")

    try:
        d, secs = complete(PORTS[a.target], prompt, a.n_predict, a.temp)
    except Exception as e:
        print(f"{a.target}: FAILED: {e}", file=sys.stderr)
        return 1

    t = d.get("timings", {})
    print(d.get("content", "").strip())
    print(f"\n--- {a.target} @ {t.get('predicted_per_second', 0):.2f} tok/s "
          f"({secs:.1f}s wall, {d.get('tokens_predicted', 0)} tokens) ---",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
