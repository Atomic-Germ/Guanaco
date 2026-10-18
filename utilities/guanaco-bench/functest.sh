#!/bin/bash
# Functional test for the cascade pair. Not efficacy: nothing here decides which
# model should get a request.
#
# Uses the server's own `timings` block rather than wall-clock arithmetic - the
# first version of this script printed epoch seconds because of a typo, and the
# host's absolute rates drift with thermal state anyway.
set -u
B=$(dirname "$(readlink -f "$0")")
RUN="$B/run"
P="$B/prompt.txt"

# ask <port> <label> <prompt> <n_predict>
ask () {
  python3 - "$1" "$2" "$3" "$4" << 'PY'
import json, sys, urllib.request
port, label, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.2, "stream": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                             data=body, headers={"Content-Type": "application/json"})
try:
    d = json.load(urllib.request.urlopen(req, timeout=900))
except Exception as e:
    print(f"  {label:6} FAILED: {e}")
    sys.exit(0)
t = d.get("timings", {})
rate = t.get("predicted_per_second", 0)
text = d.get("content", "").strip().replace("\n", " ")[:110]
print(f"  {label:6} {rate:5.2f} tok/s  {text}")
PY
}

echo "=== 1. endpoints ==="
for p in 8081 8082; do
  printf "  :%s %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:$p/health)"
done

echo
echo "=== 2. both answer the same question ==="
Q="In one sentence: what makes a consensus protocol safe when messages are lost?"
ask 8081 small "$Q" 60
ask 8082 large "$Q" 60

echo
echo "=== 3. small model, large one resident (3 rounds) ==="
LONG=$(cat "$P")
for i in 1 2 3; do
  ask 8081 small "$LONG" 120
done

echo
echo "=== 4. switching ==="
ask 8081 small "Name one invariant of Raft." 30
ask 8082 large "Name one invariant of Raft." 30
ask 8081 small "Name one invariant of Paxos." 30

echo
echo "=== state ==="
for f in small large; do
  printf "  %-6s " $f
  ps -o stat=,rss=,etimes= -p "$(cat $RUN/$f.pid)" 2>/dev/null || echo "GONE"
done
grep -E "resident subset" "$RUN/large.log" 2>/dev/null | tail -1
free -g | head -2
echo "FUNCTIONAL TEST DONE"
