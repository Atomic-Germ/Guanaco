#!/usr/bin/env python3
"""Run a benchmark cold, and report what it actually did.

Two things have gone wrong with Guanaco benchmarks on this machine:

  - results were compared across runs whose page cache state differed, so the
    I/O-bound settings had nothing to bind against and every knob looked the same;
  - "it was fast" was taken to mean the pinning was working, when it could just
    as easily mean the model was already in RAM.

So: drop the model's page cache first, poll /proc/<pid>/io while the child runs,
and report the real disk reads. A run that read ~0 bytes was not measuring
streaming and its numbers mean nothing regardless of how good they look.

usage: bench.py --model M [--drop FILE] --label NAME -- <cmd> [args...]
"""
import argparse
import os
import re
import subprocess
import sys
import time


def cached_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0


def drop_cache(paths):
    before = cached_kb()
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError as e:
            print(f"  dropcache: {p}: {e}", file=sys.stderr)
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    after = cached_kb()
    print(f"  cache: Cached {before} -> {after} kB (dropped {before - after} kB)")


def read_io(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                out[k] = int(v)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("no command", file=sys.stderr)
        return 1

    print(f"=== {a.label or ' '.join(cmd[:2])} ===", flush=True)
    drop_cache(a.drop or ([a.model] if a.model else []))

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    peak = {"read_bytes": 0, "rchar": 0}
    peak_rss = 0
    while proc.poll() is None:
        io = read_io(proc.pid)
        for k in peak:
            peak[k] = max(peak[k], io.get(k, 0))
        try:
            with open(f"/proc/{proc.pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_rss = max(peak_rss, int(line.split()[1]))
        except OSError:
            pass
        time.sleep(0.4)
    out = proc.stdout.read()
    proc.wait()
    elapsed = time.time() - t0

    tps = [float(m) for m in re.findall(r"Generation:\s*([\d.]+)\s*t/s", out)]
    notes = ["  " + l.strip() for l in out.splitlines()
             if any(k in l for k in ("resident subset", "pin cache:", "exceeds budget",
                                     "hybrid mode", "budget not set", "dense budget",
                                     "No MoE experts", "enabling Guanaco",
                                     "using plain mmap", "forcing n_gpu"))]

    gen = f"{tps[-1]:.2f}" if tps else "n/a"
    cold = "COLD" if peak["read_bytes"] > 64 << 20 else "WARM"
    print(f"  gen={gen} t/s  wall={elapsed:.0f}s  peakRSS={peak_rss/1e6:.1f}GB  "
          f"disk_read={peak['read_bytes']/1e6:.0f}MB [{cold}]")
    for l in notes[:6]:
        print(l)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
