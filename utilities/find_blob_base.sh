#!/usr/bin/env bash
# Find the newest upstream commit whose files match the pre-image blob of every
# patch in patches/*.patch. The `index <pre>..<post>` line in each patch header
# names the blob the patch was generated against, so a commit whose tree holds
# all of those blobs IS the base the patchset was made from.
set -uo pipefail

REPO="$1"
PATCHES="$2"
SINCE="${3:-2026-09-15}"

cd "$REPO" || exit 1

PATHS=()
PRES=()

for pf in "$PATCHES"/*.patch; do
    path=$(sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "$pf" | head -1)
    pre=$(sed -n 's|^index \([0-9a-f]\{4,\}\)\.\..*|\1|p' "$pf" | head -1)
    if [ -z "$path" ] || [ -z "$pre" ]; then
        echo "PARSE FAILURE: $pf (path='$path' pre='$pre')" >&2
        exit 2
    fi
    PATHS+=("$path")
    PRES+=("$pre")
done

n=${#PATHS[@]}
if [ "$n" -lt 1 ]; then
    echo "no patches parsed" >&2
    exit 2
fi
echo "tracking $n patched files across $SINCE..HEAD"

total=0
for c in $(git rev-list --since="$SINCE" HEAD); do
    total=$((total + 1))
    missing=""
    for i in $(seq 0 $((n - 1))); do
        want=${PRES[$i]}
        have=$(git rev-parse --verify -s "$c:${PATHS[$i]}" 2>/dev/null)
        if [ -z "$have" ]; then
            missing="$missing ${PATHS[$i]}(absent)"
        elif [ "${have#$want}" = "$have" ]; then
            missing="$missing ${PATHS[$i]}"
        fi
    done
    if [ -z "$missing" ]; then
        echo "EXACT-BASE $c $(git log -1 --format='%ci %s' "$c")"
        exit 0
    fi
done
echo "scanned $total commits, no commit matches every pre-image blob"
exit 1