#!/usr/bin/env bash
# Find the newest upstream commit at which every patches/*.patch applies
# with ZERO hunk offset and zero fuzz (i.e. the commit the patchset was
# generated from). Uses an alternate index so no working tree is touched.
set -uo pipefail

REPO="$1"          # upstream clone
PATCHES="$2"       # directory of patches
MAX_COMMITS="${3:-3000}"

export GIT_INDEX_FILE="$REPO/.pin-search-index"
cd "$REPO"

mapfile -t patches < <(ls "$PATCHES"/*.patch)

n=0
for c in $(git rev-list --max-count="$MAX_COMMITS" HEAD); do
    n=$((n + 1))
    git read-tree "$c" || exit 1

    ok=1
    offset=0
    for p in "${patches[@]}"; do
        out=$(git apply --cached --check -p1 --verbose "$PATCHES/$(basename "$p")" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then ok=0; break; fi
        if grep -qi 'offset' <<<"$out"; then offset=1; fi
    done

    short=$(git rev-parse --short "$c")
    if [ $ok -eq 1 ] && [ $offset -eq 0 ]; then
        echo "EXACT  $c $short $(git log -1 --format=%ci "$c") $(git log -1 --format=%s "$c")"
        exit 0
    elif [ $ok -eq 1 ]; then
        echo "applies-with-offset  $short $(git log -1 --format=%ci "$c")"
    else
        echo "does-not-apply       $short $(git log -1 --format=%ci "$c")"
    fi
done

rm -f "$GIT_INDEX_FILE"
echo "NO EXACT BASE FOUND in $MAX_COMMITS commits"
exit 1