#!/usr/bin/env python3
"""Load the equal-size specialist/generalist pair for the domain-routing test.

  :8081 generalist   Qwen3.8-4B-Distill Q8_0        4.61 GB
  :8082 specialist  Electrical_Engineering_Specialist 4.06 GB (Phi3-4B)

Both CPU-bound (-ngl 0) because llama-server segfaults on the second request
whenever the KV cache is in a Vulkan buffer, and these are the same size class
so there is no throughput difference to protect anyway.
"""
import os
import subprocess
import sys
import time
import urllib.request

BIN = "/home/atomic-germ/Projects/Guanaco-Test/build/bin"
HOME = os.path.expanduser("~/guanaco-bench")

ROLE = {
    "generalist": (f"{HOME}/model_4b.txt", 8081),
    "specialist": (f"{HOME}/model_ee.txt", 8082),
}


def path_of(spec):
    with open(spec) as f:
        return f.read().strip()


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start(role):
    spec, port = ROLE[role]
    if healthy(port):
        return True
    env = dict(os.environ)
    for k in ("LLAMA_ARG_LOAD_MODE", "LLAMA_ARG_GPU_PILL"):
        env.pop(k, None)
    log = open(f"{HOME}/run/{role}.log", "a")
    subprocess.Popen(
        [f"{BIN}/llama-server", "-m", path_of(spec), "--port", str(port),
         "-ngl", "0", "-c", "4096", "-t", "6", "--no-warmup"],
        env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True)
    for _ in range(180):
        if healthy(port):
            print(f"  {role} ready")
            return True
        time.sleep(2)
    print(f"  {role} TIMED OUT - see {HOME}/run/{role}.log")
    return False


if __name__ == "__main__":
    os.makedirs(f"{HOME}/run", exist_ok=True)
    ok = all(start(r) for r in sys.argv[1:] or ["generalist", "specialist"])
    sys.exit(0 if ok else 1)
