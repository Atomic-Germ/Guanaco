#!/usr/bin/env bash
# Find the newest upstream commit matching ALL of:
#   - the pre-image blob of every patch in patches/*.patch
#   - the current content of extra local files (by content hash)
set -uo pipefail
REPO="$1"; PATCHES="$2"; USER_TREE="$3"; SINCE="${4:-2026-08-01}"
cd "$REPO" || exit 1

PATHS=(); HASHES=(); SOURCES=()
for pf in "$PATCHES"/*.patch; do
    PATHS+=("$(sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "$pf" | head -1)")
    HASHES+=("$(sed -n 's|^index \([0-9a-f]\{4,\}\)\.\..*|\1|p' "$pf" | head -1)")
    SOURCES+=("patch:$(basename "$pf")")
done
# extra constraints: files where the local tree diverges from base+patches
for f in ggml/src/CMakeLists.txt \
         ; do
    if [ -f "$USER_TREE/$f" ]; then
        PATHS+=("$f")
        HASHES+=("$(git hash-object "$USER_TREE/$f")")
        SOURCES+=("local:$f")
    fi
done

echo "constraints: ${#PATHS[@]} files"
n=${#PATHS[@]}
for c in $(git rev-list --since="$SINCE" HEAD); do
    missing=""
    for i in $(seq 0 $((n - 1))); do
        have=$(git rev-parse --verify -s "$c:${PATHS[$i]}" 2>/dev/null)
        want=${HASHES[$i]}
        if [ -z "$have" ]; then missing="$missing ${SOURCES[$i]}(absent)"
        elif [ "${have#$want}" = "$have" ]; then missing="$missing ${SOURCES[$i]}"; fi
    done
    if [ -z "$missing" ]; then
        echo "MATCH $c $(git log -1 --format='%ci %s' "$c")"
        exit 0
    fi
done
echo "no commit matches all constraints"
exit 1