#!/usr/bin/env python3
"""Is escalation better than giving the quick model more reasoning effort?

This is the control the cascade needs. If the quick model can be made as good by
thinking harder, escalating buys nothing and the whole idea is redundant.

Measured on the roles:
  quick      AesCode-8B          reasoning_effort accepted but INERT (no thinking
                                 block at any level; identical answers low vs max)
  considered Qwen3.8-27B dense   real thinking, supports low/medium/xhigh with
                                 xhigh as the DEFAULT

So on this pairing the comparison is asymmetric by construction: the considered
model already thinks as hard as it can, and the quick model cannot be made to
think at all. What is left to measure is whether that trade is worth it - which
needs questions where reasoning changes the answer, and a time cost to weigh
against the quality gain.

Questions have definite, checkable answers so grading is not a judgement call.
"""
import json
import sys
import time
import urllib.request

QUICK = "http://127.0.0.1:8081"
CONSIDERED = "http://127.0.0.1:8082"

# (question, expected substring, difficulty note)
QUESTIONS = [
    ("A bat and a ball cost $1.10 together. The bat costs $1.00 more than the "
     "ball. How much does the ball cost? Answer with the number only.", "0.05",
     "classic trap; wrong answer is 0.10"),

    ("What comes next: 1, 11, 21, 1211, 111221, ? Answer with the number only.",
     "312211", "look-and-say; needs the rule found, not guessed"),

    ("Cheryl's birthday puzzle. Albert: I don't know, but I know Bernard "
     "doesn't know. Bernard: at first I didn't know, now I know. "
     "Albert: then I also know. Cheryl is one of (May 15, May 16, May 19, "
     "June 17, June 18). What is the date? Answer with the date only.",
     "july 16", "multi-step epistemic reasoning"),

    ("A snail climbs a 10 m well. Each day it climbs 3 m, each night it slips "
     "back 2 m. On which day does it reach the top? Answer with the day number "
     "only.", "8", "off-by-one; the classic trap is 10"),

    ("If a train travels 60 km in 45 minutes, how long does it take to travel "
     "100 km at the same speed? Answer in minutes only.", "75",
     "rate reasoning; the trap is 100"),

    ("I have 3 boxes. Box 1 is labelled 'apples'. Box 2 is labelled 'oranges'. "
     "Box 3 is labelled 'apples and oranges'. All labels are wrong. I may draw "
     "one fruit from one box. Which box do I choose, and why?",
     "3", "classic mislabel puzzle; needs the constraint applied"),

    ("Write a Python function that returns the n-th Fibonacci number using "
     "iteration, not recursion.", "def ", "code generation"),

    ("A train 100 m long travels at 60 km/h. How far does it travel while a "
     "person takes 10 seconds to walk across the track at 3 km/h? Answer in "
     "metres only.", "175", "compound relative-motion reasoning"),
]


def ask(port, prompt, effort, max_tokens=700):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "reasoning_effort": effort,
    }).encode()
    req = urllib.request.Request(
        f"{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    secs = time.time() - t0
    m = d["choices"][0]["message"]
    return {
        "content": (m.get("content") or "").strip(),
        "reasoning": m.get("reasoning_content") or "",
        "secs": secs,
        "tokens": d["usage"]["completion_tokens"],
    }


def grade(ans, expect):
    a = ans.lower()
    if not ans:
        return False
    if expect in a:
        return True
    # tolerate a leading article or trailing punctuation
    return expect in a.replace("the ", "").replace(".", "").replace(",", "")


def main():
    configs = [
        ("quick",      QUICK, "low"),
        ("quick",      QUICK, "max"),
        ("considered", CONSIDERED, "low"),
        ("considered", CONSIDERED, "xhigh"),
    ]
    roles = sys.argv[1:] or ["quick:low", "quick:max",
                             "considered:low", "considered:xhigh"]
    configs = [(r.split(":")[0],
                QUICK if r.startswith("quick") else CONSIDERED,
                r.split(":")[1]) for r in roles]

    print(f"{'#':<3}{'question':<14}" + "".join(f"{r.replace(':', '/'):>22}"
                                               for r in roles))
    print("-" * (17 + 22 * len(roles)))
    totals = {f"{r[0]}:{r[2]}": [0, 0.0, 0] for r in configs}
    for qi, (q, expect, _note) in enumerate(QUESTIONS, 1):
        row = f"{qi:<3}{expect:<14}"
        for role, port, effort in configs:
            key = f"{role}:{effort}"
            try:
                r = ask(port, q, effort)
                ok = grade(r["content"], expect)
                totals[key][0] += 1 if ok else 0
                totals[key][1] += r["secs"]
                totals[key][2] += r["tokens"]
                mark = "OK " if ok else "BAD"
                row += f"{mark}{r['secs']:6.1f}s{r['tokens']:5d}t  "
            except Exception as e:
                row += f"{'ERR':<18}"
            sys.stdout.write("\r" + row)
            sys.stdout.flush()
        print(row)
    print("\n=== totals ===")
    print(f"{'config':<22}{'correct':>9}{'time':>10}{'tokens':>9}")
    for k, (c, s, t) in totals.items():
        print(f"{k:<22}{c}/{len(QUESTIONS):<7}{s:8.0f}s{t:8d}")


if __name__ == "__main__":
    main()
