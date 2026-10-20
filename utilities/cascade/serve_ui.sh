#!/bin/bash
# Restart the cascade UI and report what is up.
set -u
B=$(dirname "$(readlink -f "$0")")
pkill -f "python3 $B/webui.py" 2>/dev/null
sleep 2
setsid python3 "$B/webui.py" > "$HOME/guanaco-bench/run/webui.log" 2>&1 < /dev/null &
sleep 4
echo "page:  $(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8090/)"
echo "status: $(curl -s --max-time 10 http://127.0.0.1:8090/status)"
echo
echo "quick model: $(python3 "$B/cascade.py" status 2>/dev/null | grep quick || echo down)"
