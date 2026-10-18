#!/usr/bin/env python3
"""Drop the page cache for a file, without root.

"Load time" measured with the model already resident in page cache is not the
number anyone actually pays, so every cold-start measurement needs this first.
posix_fadvise(POSIX_FADV_DONTNEED) evicts a file's clean pages; no privileges
needed. Reports the before/after so the drop is verified rather than assumed.
"""
import os
import sys


def cached_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0


def main():
    if len(sys.argv) < 2:
        print("usage: dropcache.py <file> [...]", file=sys.stderr)
        return 1
    before = cached_kb()
    for path in sys.argv[1:]:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    after = cached_kb()
    print(f"Cached: {before} kB -> {after} kB (dropped {before - after} kB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
