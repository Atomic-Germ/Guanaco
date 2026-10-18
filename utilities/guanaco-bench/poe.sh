#!/bin/bash
# Bring up the pair and run the functional test, CPU-bound.
set -u
B=$(dirname "$(readlink -f "$0")")

echo "########## starting pair (large = dense) ##########"
"$B/pair.sh" start dense 2>&1 | tail -3
"$B/pair.sh" status

echo
echo "########## functional test ##########"
bash "$B/functest.sh" 2>&1

echo
echo "########## pair still alive? ##########"
"$B/pair.sh" status
echo "POE DONE"
