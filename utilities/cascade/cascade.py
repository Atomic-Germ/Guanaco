#!/usr/bin/env python3
"""Cascade: a quick model for most work, a considered model for the rest.

TERMINOLOGY. The two roles are quick and considered, not small and large.
Parameter count does no work in this design. A 24B-A8B MoE has 8B active, runs
3.3x faster than a 27B dense, and is less capable one-shot - so it is the quick
model despite being the larger file. Capability and cost-per-token are
independent axes, and a model is assigned a role explicitly rather than
inferred from its size.

This is NOT speculative decoding. That trades samples for speed: a cheap model
drafts, an expensive model verifies, and the same answer set comes out faster.
This trades samples for capability: a cheap model answers, and occasionally the
answer is replaced by one from a better model. The cascade makes ONE escalation
decision per request and gets no second sample, which is why "more correct
one-shot" is the property the considered model needs.

MEASURED, on this machine (see docs/OPERATING_CHARACTER.md and the notes below):

    quick      AesCode-8B          ~6.4 tok/s   generic
    considered Qwen3.8-27B dense   ~1.3 tok/s   more capable one-shot
               Huihui-MoE-24B-A8B  ~4.4 tok/s   fast but less capable

The considered model's cost is the whole point of the switch:

    resident    considered always loaded; quick stays ~6.4 tok/s
    on-demand   considered loads per use (~20 s warm, ~40-70 s cold)

on-demand wins unless the considered model is used more than about once per
40-130 quick tokens. That threshold is a property of the user's workload, not
something this tool can decide.

BOTH ROLES RUN CPU-BOUND (-ngl 0). llama-server with any Vulkan-backed KV
cache segfaults on the second request in ggml_vk_buffer_read_2d, reached from
server_slot::prompt_save -> state_seq_get_data -> llama_io_write_host.
Reproduced 4/4 with identical stacks, with gpu-pill on AND off, so it is not
Guanaco's UMA allocation. Cost is ~6.4 vs ~7.6 tok/s for the quick model.

usage:
  cascade.py up                     start the quick model
  cascade.py ask QUICK "prompt"     ask the quick model
  cascade.py ask CONSIDERED "..."   ask the considered model (loads it if needed)
  cascade.py serve                  interactive loop
  cascade.py status                 what is running
  cascade.py down                   stop everything
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN = "/home/atomic-germ/Projects/Guanaco-Test/build/bin"
HOME = os.path.expanduser("~/guanaco-bench")

QUICK_PORT = 8081
CONSIDERED_PORT = 8082

# role -> (model file, port, extra env, extra args)
MODELS = {
    "quick": {
        "file": f"{HOME}/model_8b.txt",
        "port": QUICK_PORT,
        "env": {},
        "args": ["-c", "4096", "-t", "6"],
    },
    # dense 27B: the measured "considered" pairing
    "considered_dense": {
        "file": f"{HOME}/model_path.txt",
        "port": CONSIDERED_PORT,
        "env": {
            "LLAMA_ARG_LOAD_MODE": "streaming",
            "LLAMA_ARG_GPU_PILL": "1",
            "LLAMA_ARG_KV_UNIFIED": "0",
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q4_0",
            "LLAMA_ARG_THREADS": "4",
            "LLAMA_ARG_GUANACO_DENSE_BUDGET": "7000",
        },
        "args": ["-c", "2048", "-t", "4"],
    },
    # MoE variant: same role, different model - demonstrates that role is not size
    "considered_moe": {
        "file": f"{HOME}/model_moe.txt",
        "port": CONSIDERED_PORT,
        "env": {
            "LLAMA_ARG_LOAD_MODE": "streaming",
            "LLAMA_ARG_GPU_PILL": "1",
            "LLAMA_ARG_KV_UNIFIED": "0",
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q4_0",
            "LLAMA_ARG_THREADS": "4",
            "LLAMA_ARG_GUANACO_DENSE_BUDGET": "7000",
        },
        "args": ["-c", "2048", "-t", "4"],
    },
}

LIVE = {}  # role -> Popen


def model_path(spec):
    with open(spec["file"]) as f:
        return f.read().strip()


def healthy(port, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start(role, wait=True):
    if healthy(MODELS[role]["port"]):
        return True
    spec = MODELS[role]
    env = dict(os.environ)
    for k in ("LLAMA_ARG_LOAD_MODE", "LLAMA_ARG_GPU_PILL"):
        env.pop(k, None)
    env.update(spec["env"])
    log = open(f"{HOME}/run/{role}.log", "a")
    cmd = ([f"{BIN}/llama-server", "-m", model_path(spec),
            "--port", str(spec["port"]), "-ngl", "0", "--no-warmup"]
           + spec["args"])
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    LIVE[role] = p
    if not wait:
        return True
    deadline = time.time() + 900
    while time.time() < deadline:
        if healthy(spec["port"]):
            return True
        if p.poll() is not None:
            print(f"  {role}: died during load, see {HOME}/run/{role}.log",
                  file=sys.stderr)
            return False
        time.sleep(2)
    print(f"  {role}: timed out loading", file=sys.stderr)
    return False


def stop(role):
    spec = MODELS[role]
    p = LIVE.pop(role, None)
    if p is not None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    subprocess.run(["pkill", "-f", f"llama-server.*{spec['port']}"],
                   capture_output=True)
    for _ in range(20):
        if not healthy(spec["port"], timeout=1):
            return
        time.sleep(0.5)


def ask(role, prompt, n_predict=160, temperature=0.2, timeout=1800):
    port = MODELS[role]["port"]
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": temperature, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d, time.time() - t0


def report(role, d, secs, show=True):
    if show:
        print(d.get("content", "").strip())
    t = d.get("timings", {})
    print(f"\n--- {role}: {t.get('predicted_per_second', 0):.2f} tok/s, "
          f"{secs:.1f}s, {d.get('tokens_predicted', 0)} tokens ---",
          file=sys.stderr)


def get(role, considered_spec, on_demand):
    """Ensure the role is up; on-demand loads the considered model on use."""
    if healthy(MODELS[role]["port"]):
        return True
    if role == "quick":
        return start("quick")
    print("  (loading the considered model - this is the on-demand cost)",
          file=sys.stderr)
    t0 = time.time()
    if not start(considered_spec):
        return False
    print(f"  (loaded in {time.time() - t0:.0f}s)", file=sys.stderr)
    if on_demand:
        # unload after this request, so the cost is paid per use rather than
        # continuously. See the module docstring for the break-even.
        return True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--considered", default="considered_dense",
                    choices=[k for k in MODELS if k != "quick"],
                    help="which model plays the considered role")
    ap.add_argument("--mode", choices=["resident", "on-demand"],
                    default="on-demand")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("up")
    sub.add_parser("down")
    sub.add_parser("status")
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("role", choices=["quick", "considered"])
    p_ask.add_argument("prompt")
    p_ask.add_argument("-n", "--n-predict", type=int, default=160)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("-n", "--n-predict", type=int, default=160)

    a = ap.parse_args()
    os.makedirs(f"{HOME}/run", exist_ok=True)
    on_demand = a.mode == "on-demand"

    if a.cmd == "up":
        ok = start("quick")
        print("quick ready" if ok else "quick FAILED")
        if not on_demand and not start(a.considered):
            print("considered FAILED")
        return 0 if ok else 1

    if a.cmd == "down":
        for r in ("quick", a.considered):
            stop(r)
        print("stopped")
        return 0

    if a.cmd == "status":
        for r in ("quick", a.considered):
            spec = MODELS[r]
            up = healthy(spec["port"])
            size = os.path.getsize(model_path(spec)) / 1e9 if up else 0
            print(f"  {r:16} :{spec['port']}  {'up' if up else 'down'}"
                  f"{f'  {size:.1f} GB' if up else ''}")
        return 0

    if a.cmd == "ask":
        role = "quick" if a.role == "quick" else a.considered
        if not get(role, a.considered, on_demand):
            return 1
        try:
            d, secs = ask(role, a.prompt, a.n_predict)
        except Exception as e:
            print(f"{role}: FAILED: {e}", file=sys.stderr)
            return 1
        report(role, d, secs)
        if on_demand and role != "quick":
            stop(a.considered)
        return 0

    if a.cmd == "serve":
        if not start("quick"):
            return 1
        print("quick:q  considered:c  quit:q")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line or line in ("quit", "q"):
                return 0
            if line[0] in ("q", "Q"):
                role, prompt = "quick", line[1:].strip()
            elif line[0] in ("c", "C"):
                role, prompt = a.considered, line[1:].strip()
            else:
                role, prompt = "quick", line
            if not prompt:
                continue
            if not get(role, a.considered, on_demand):
                continue
            try:
                d, secs = ask(role, prompt, a.n_predict)
                report(role, d, secs)
            except Exception as e:
                print(f"{role}: FAILED: {e}", file=sys.stderr)
            if on_demand and role != "quick":
                stop(a.considered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
