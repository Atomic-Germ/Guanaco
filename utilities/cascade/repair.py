#!/usr/bin/env python3
"""Cascade as repair: quick attempts, the verifier decides, considered fixes.

The factual-correctness framing asks the quick model to *predict* whether its
answer was good enough, which is not something it can do. Repair inverts that:
attempt, then let an objective verifier decide, and escalate only on verifiable
failure. No quality model is needed because the build output already is one.

    quick writes code -> run tests
        pass -> done
        fail -> considered rewrites, given the actual error -> run tests again

Each task here is a function plus assertions. A solution counts only if it
passes every assertion, so "shallow but plausible" is caught rather than
rewarded - which is the failure mode a small model actually has.

Reported per task:
  quick   pass on first attempt, and how long that took
  repair  whether the considered model fixed it, and total time for the pair

usage: repair.py [--quick quick|c4b] [--only N]
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

QUICK = "http://127.0.0.1:8081"
CONSIDERED = "http://127.0.0.1:8082"
ALTERNATE = "http://127.0.0.1:8083"

# Each task: prompt, and a self-contained test harness appended to the response.
# The test runs in a subprocess with a timeout, so a wrong model that loops
# cannot hang the run.
TASKS = [
    dict(
        name="binary_search",
        prompt="Write a Python function `binary_search(arr, target)` that returns "
               "the index of target in the sorted list arr, or -1 if absent. "
               "Return only the function definition.",
        tests="""
assert binary_search([1,3,5,7,9], 5) == 3
assert binary_search([1,3,5,7,9], 1) == 0
assert binary_search([1,3,5,7,9], 9) == 4
assert binary_search([], 1) == -1
assert binary_search([2,4], 3) == -1
assert binary_search([2], 2) == 0
""",
    ),
    dict(
        name="merge_sorted",
        prompt="Write a Python function `merge_sorted(a, b)` that merges two "
               "sorted lists into one sorted list. Return only the function.",
        tests="""
assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
assert merge_sorted([], [1,2]) == [1,2]
assert merge_sorted([1,2], []) == [1,2]
assert merge_sorted([], []) == []
assert merge_sorted([1,1,3],[1,2]) == [1,1,1,2,3]
""",
    ),
    dict(
        name="balanced_brackets",
        prompt="Write a Python function `is_balanced(s)` returning True if the "
               "bracket string is balanced, ignoring characters that are not "
               "brackets. Return only the function.",
        tests="""
assert is_balanced("()[]{}") is True
assert is_balanced("([{}])") is True
assert is_balanced("(") is False
assert is_balanced(")") is False
assert is_balanced("([)]") is False
assert is_balanced("") is True
assert is_balanced("a(b)c") is True
""",
    ),
    dict(
        name="lru_cache",
        prompt="Write a Python class `LRU` with `get(key)` and `put(key, value)`. "
               "Capacity is 2. `get` returns None on miss. Eviction is "
               "least-recently-used. Return only the class.",
        tests="""
c = LRU(2)
c.put('a',1); c.put('b',2)
assert c.get('a') == 1
c.put('c',3)
assert c.get('b') is None, "b should have been evicted"
assert c.get('a') == 1
assert c.get('c') == 3
c.put('d',4)
assert c.get('a') is None
""",
    ),
    dict(
        name="rotate_matrix",
        prompt="Write a Python function `rotate(matrix)` that rotates an NxN "
               "matrix 90 degrees clockwise, in place, and returns it. "
               "Return only the function.",
        tests="""
assert rotate([[1,2],[3,4]]) == [[3,1],[4,2]]
assert rotate([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]
assert rotate([[1]]) == [[1]]
""",
    ),
    dict(
        name="word_frequency",
        prompt="Write a Python function `word_freq(text)` returning a dict "
               "mapping each lowercase word to its count. Split on whitespace "
               "and strip punctuation ,.!? from the ends of words. Return only "
               "the function.",
        tests="""
assert word_freq("the cat the dog") == {'the':2, 'cat':1, 'dog':1}
assert word_freq("") == {}
assert word_freq("a a a") == {'a':3}
assert word_freq("Hello, hello! world?") == {'hello':2, 'world':1}
""",
    ),
    # --- pygame ---
    # Small models reproducibly forget pygame.init(). This is not a guess: it is
    # an observed failure mode, and it is a good basis for this harness for two
    # reasons - it is common, and real pygame fails fast and loudly when the
    # video system was never initialised.
    #
    # The stub below reproduces that exact behaviour (set_mode raises unless
    # init() was called) instead of requiring pygame to be installed, so the
    # test is fast, deterministic, and fails for the same reason real code does.
    # Models appear to pattern-match the visible shape of pygame code - the
    # import is salient, the init call is not - and reproduce the shape while
    # skipping the setup the surrounding context used to provide.
    dict(
        name="pygame_init_window",
        prompt="Write a Python program that opens a 640x480 pygame window titled "
               "'Demo' and fills it with blue. Produce a complete, runnable "
               "program.",
        pre="""
import pygame
pygame.error = type('error', (Exception,), {})
_STATE = {'inited': False}
pygame.init = lambda: _STATE.__setitem__('inited', True)
def _fake_set_mode(size=0, flags=0, depth=0):
    if not _STATE['inited']:
        raise pygame.error("video system not initialized")
    return object()
pygame.display.set_mode = _fake_set_mode
pygame.display.set_caption = lambda *a, **k: None
pygame.draw.rect = lambda *a, **k: None
pygame.quit = lambda: None
pygame.Color = lambda *a, **k: (0, 0, 255)
""",
        tests="""
pygame.init()
pygame.display.set_mode((640, 480))
pygame.display.set_caption('Demo')
pygame.quit()
""",
        check="init_called",
    ),
    dict(
        name="pygame_sprite_loop",
        prompt="Write a Python program using pygame that creates a Sprite "
               "subclass for a player, makes a screen, and runs a main loop "
               "with a clock at 60 FPS. Produce a complete, runnable program.",
        pre="""
import pygame
pygame.error = type('error', (Exception,), {})
_STATE = {'inited': False}
pygame.init = lambda: _STATE.__setitem__('inited', True)
class _FakeClock:
    def tick(self, framerate=0): pass
pygame.time = type('time', (), {})()
pygame.time.Clock = lambda: _FakeClock()
def _fake_set_mode(size=0, flags=0, depth=0):
    if not _STATE['inited']:
        raise pygame.error("video system not initialized")
    return type('S', (), {'blit': lambda *a, **k: None, 'fill': lambda *a, **k: None})()
pygame.display.set_mode = _fake_set_mode
pygame.display.set_caption = lambda *a, **k: None
pygame.Sprite = type('Sprite', (), {})
pygame.sprite = type('sprite', (), {})()
pygame.quit = lambda: None
""",
        tests="""
pygame.init()
screen = pygame.display.set_mode((800, 600))
class Player(pygame.sprite.Sprite):
    pass
clock = pygame.time.Clock()
clock.tick(60)
pygame.quit()
""",
        check="init_called",
    ),
]


def ask(port, prompt, max_tokens=900, temperature=0.2, effort=None):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        f"{port}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return d["choices"][0]["message"].get("content") or "", time.time() - t0


def extract_code(text):
    """Pull function/class definitions out of a chat answer."""
    # fenced block if present, else the whole thing - both models vary
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[-1] if m else text
    keep = []
    for block in re.split(r"\n(?=\S)", code):
        if re.match(r"\s*(def |class |import |from )", block):
            keep.append(block)
    return "\n\n".join(keep) if keep else code


def run_tests(code, tests, pre="", check="", timeout=25):
    """Return (passed, error_text). Runs in a subprocess so a hang is bounded.

    `pre` is prepended (the pygame stub). `check` selects how completion is
    judged. For the pygame tasks the stub itself is the test: executing the
    model's code calls set_mode, which raises "video system not initialized"
    unless the program called init() first - exactly the real failure. The test
    must therefore NOT call init() itself, or it would pass regardless of what
    the model wrote.
    """
    if check == "init_called":
        # The model often guards its work behind main(). Trigger it, then let the
        # stub's set_mode decide whether init happened first.
        tail = """
try:
    main()
except (NameError, TypeError):
    pass
except Exception:
    pass
assert _STATE['inited'], "pygame.init() was never called"
print('ALL_PASS')
"""
        src = pre + "\n" + code + "\n\n" + tail
    else:
        src = pre + "\n" + code + "\n\n" + tests + "\nprint('ALL_PASS')\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        ok = p.returncode == 0 and "ALL_PASS" in p.stdout
        err = "" if ok else (p.stderr.strip().splitlines() or ["failed"])[-1]
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "timed out (possible infinite loop)"
    finally:
        import os
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int)
    ap.add_argument("--quick-port", default=QUICK)
    ap.add_argument("--considered-port", default=CONSIDERED)
    a = ap.parse_args()
    quick, considered = a.quick_port, a.considered_port

    tasks = TASKS if a.only is None else [TASKS[a.only]]
    print(f"{'task':<20}{'quick':>10}{'t':>8}{'repaired':>10}{'total':>9}")
    print("-" * 57)
    stats = {"n": 0, "quick_pass": 0, "repaired": 0, "quick_t": 0.0, "tot_t": 0.0}

    for t in tasks:
        stats["n"] += 1
        code, t1 = ask(quick, t["prompt"])
        ok1, err = run_tests(extract_code(code), t["tests"],
                             t.get("pre", ""), t.get("check", ""))
        stats["quick_t"] += t1
        if ok1:
            stats["quick_pass"] += 1
            stats["tot_t"] += t1
            print(f"{t['name']:<20}{'PASS':>10}{t1:7.1f}s{'-':>10}{t1:8.1f}s")
            continue

        stats["tot_t"] += t1
        repair_prompt = (
            f"{t['prompt']}\n\n"
            f"Your previous attempt failed with:\n{err}\n\n"
            "Return only the corrected function.")
        code2, t2 = ask(considered, repair_prompt)
        ok2, err2 = run_tests(extract_code(code2), t["tests"],
                               t.get("pre", ""), t.get("check", ""))
        stats["repaired"] += 1 if ok2 else 0
        stats["tot_t"] += t2
        print(f"{t['name']:<20}{'FAIL':>10}{t1:7.1f}s"
              f"{('PASS' if ok2 else 'FAIL'):>10}{t1 + t2:8.1f}s"
              f"   [{err[:34]}]")

    n = stats["n"]
    print(f"\nfirst-attempt pass: {stats['quick_pass']}/{n}"
          f"   repaired: {stats['repaired']}/{n - stats['quick_pass']}"
          f"   final: {stats['quick_pass'] + stats['repaired']}/{n}")
    print(f"time: quick {stats['quick_t']:.0f}s, "
          f"total including repair {stats['tot_t']:.0f}s")


if __name__ == "__main__":
    main()
