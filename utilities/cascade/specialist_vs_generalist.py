#!/usr/bin/env python3
"""Does domain specialisation at equal size beat a generalist on its own domain?

The small/considered cascade pays for escalation in throughput: the better model
is slower, so every escalation costs time. Specialist/generalist does not. Two
models of the same size run at the same rate, so choosing between them costs
nothing but memory - the question is purely which one is more right.

  generalist   Qwen3.8-4B-Distill Q8_0   4.61 GB
  specialist  Electrical_Engineering_Specialist Q8_0 (Phi3-4B)   4.06 GB

Same size class, different architecture, different training. The design is a 2x2:

  EE question    -> who wins?
  non-EE question -> who wins?

If the specialist wins its domain and loses outside it, routing by domain is
worth doing and needs no quality prediction - only classification, which is a far
tractable problem. If it wins everywhere it is simply a better model and the
"specialist" framing is wrong.

Questions have definite numeric answers so grading is mechanical.
"""
import json
import sys
import time
import urllib.request

GENERALIST = "http://127.0.0.1:8081"
SPECIALIST = "http://127.0.0.1:8082"

EE = [
    ("A 12 V source drives a 470 ohm resistor. What current flows? "
     "Answer in milliamps, number only.", "40.7"),
    ("What is the time constant of a 10 uF capacitor and a 1 kohm resistor, "
     "in milliseconds? Number only.", "10"),
    ("Two 1 kohm resistors in parallel. What is the equivalent resistance, in "
     "ohms? Number only.", "500"),
    ("A 2 kohm resistor dissipates 0.5 W. What is the voltage across it, in "
     "volts? Number only.", "31.6"),
    ("What charge, in microCoulombs, does a 100 nF capacitor hold at 5 V? "
     "Number only.", "0.5"),
    ("At 50 Hz, what is the reactance of a 10 uF capacitor in ohms? "
     "Number only.", "318"),
    ("A 100 W lamp runs for 8 hours. How many joules of energy does it use, in "
     "kJ? Number only.", "2880"),
    ("What is the voltage gain magnitude of an op-amp in a non-inverting "
     "configuration with Rf = 10k and Rg = 1k? Number only.", "11"),
]

NON_EE = [
    ("A bat and a ball cost $1.10 together. The bat costs $1.00 more than the "
     "ball. How much does the ball cost? Answer with the number only.", "0.05"),
    ("What comes next: 1, 11, 21, 1211, 111221, ? Answer with the number only.",
     "312211"),
    ("A snail climbs a 10 m well, climbing 3 m by day and slipping 2 m at "
     "night. On which day does it reach the top? Number only.", "8"),
    ("A train travels 60 km in 45 minutes. How long for 100 km at the same "
     "speed, in minutes? Number only.", "75"),
    ("In Python, what does list(range(3, 0, -1)) evaluate to? Answer with the "
     "list only.", "[3, 2, 1]"),
    ("Which planet has the shortest day in the solar system? Answer with the "
     "name only.", "jupiter"),
    ("Write a Python function that reverses a string. Return only the "
     "function.", "def "),
    ("A train 100 m long travels at 60 km/h. How far does it travel while a "
     "person takes 10 s to cross the track at 3 km/h, in metres? Number only.",
     "175"),
]


def ask(port, prompt, max_tokens=500):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return (d["choices"][0]["message"].get("content") or ""), time.time() - t0


def grade(text, expect):
    t = (text or "").lower().replace(",", "").replace(" ", "")
    e = expect.lower().replace(",", "").replace(" ", "")
    return e in t


def run_set(name, items):
    print(f"\n=== {name} ===")
    print(f"{'expected':<12}{'generalist':<24}{'specialist':<24}")
    g_ok = s_ok = 0
    gt = st = 0.0
    for q, expect in items:
        gc, gsec = ask(GENERALIST, q)
        sc, ssec = ask(SPECIALIST, q)
        g = grade(gc, expect)
        s = grade(sc, expect)
        g_ok += g
        s_ok += s
        gt += gsec
        st += ssec
        print(f"{expect:<12}{('OK ' if g else 'BAD') + f' {gsec:5.1f}s':<24}"
              f"{('OK ' if s else 'BAD') + f' {ssec:5.1f}s':<24}")
    n = len(items)
    print(f"{'TOTAL':<12}{f'{g_ok}/{n}  {gt:5.0f}s':<24}{f'{s_ok}/{n}  {st:5.0f}s':<24}")
    return g_ok, s_ok, n


def main():
    print("model roles:  :8081 = generalist (Qwen3.8-4B)   "
          ":8082 = specialist (EE, Phi3-4B)")
    ge, se, n = run_set("electrical engineering (the specialist's domain)", EE)
    gn, sn, _ = run_set("non-EE (the generalist's strength)", NON_EE)

    print("\n=== summary ===")
    print(f"{'':<14}{'generalist':>12}{'specialist':>12}")
    print(f"{'EE':<14}{f'{ge}/{n}':>12}{f'{se}/{n}':>12}")
    print(f"{'non-EE':<14}{f'{gn}/{n}':>12}{f'{sn}/{n}':>12}")
    print(f"{'total':<14}{f'{ge+gn}/{2*n}':>12}{f'{se+sn}/{2*n}':>12}")
    dom = (se - ge) / n
    off = (sn - gn) / n
    print(f"\nadvantage on the specialist's own domain: {dom:+.2f} per task")
    print(f"advantage outside it:                     {off:+.2f} per task")
    if dom > 0.15 and off < 0:
        print("\n-> specialist wins its domain and loses outside it: "
              "domain routing is worth doing, and needs classification "
              "rather than quality prediction.")
    elif dom > 0.15 and off > 0:
        print("\n-> specialist wins everywhere: it is simply a better model. "
              "The specialist framing is wrong.")
    else:
        print("\n-> no clean specialisation signal in this sample.")


if __name__ == "__main__":
    main()
