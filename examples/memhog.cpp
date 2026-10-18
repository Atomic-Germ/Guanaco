// Memory pressure harness for the big/small cascade work.
//
// The concurrency test showed the 8B dropping from 7.4 to 5.5 tok/s while the
// 27B generated, but that confounds several things: a second GPU workload, disk
// traffic, page cache eviction, and CPU contention. A second model cannot
// separate them.
//
// This sits there and does exactly one thing, so each mode isolates one suspect:
//
//   --alloc N     allocate and touch N GiB, then idle. Pure memory pressure.
//
//   --stream N R  allocate N GiB, then re-read it in a loop targeting R MB/s.
//                 This is what a pager looks like to the rest of the system:
//                 constant traffic through the same memory bus, on UMA, where
//                 the iGPU and CPU are also reading from it.
//
// The UMA point is the interesting one. If --stream hurts and --alloc does not,
// the cost is memory bandwidth shared with the GPU, not capacity and not CPU -
// which would mean the cascade needs bandwidth reservation rather than RAM
// reservation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

#include <sys/mman.h>

int main(int argc, char ** argv) {
    size_t gb = 24;
    bool stream = false;
    double rate_mbs = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--alloc") && i + 1 < argc) {
            gb = (size_t) atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--stream") && i + 1 < argc) {
            gb = (size_t) atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--rate") && i + 1 < argc) {
            rate_mbs = atof(argv[++i]);
            stream = true;
        } else if (!strcmp(argv[i], "--help")) {
            printf("usage: memhog [--alloc N | --stream N] [--rate MB/s]\n");
            return 0;
        }
    }

    const size_t n = gb << 30;
    // MAP_NORESERVE so the allocation itself cannot fail under a cgroup limit;
    // the point is to occupy the pages, and the kernel reclaims under pressure.
    char * p = (char *) mmap(nullptr, n, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "memhog: mmap of %zu GiB failed\n", gb);
        return 1;
    }

    // Touch every page so the memory is really resident, not just reserved.
    const size_t page = 4096;
    for (size_t off = 0; off < n; off += page) {
        p[off] = 1;
    }
    fprintf(stderr, "memhog: %zu GiB touched\n", gb);
    fflush(stderr);

    if (!stream) {
        // Idle holding the memory. Signal readiness and wait to be killed.
        while (true) {
            std::this_thread::sleep_for(std::chrono::hours(1));
        }
    }

    // Streaming mode: re-read the buffer at a target rate. Reads are summed so
    // the compiler cannot elide them.
    volatile unsigned long sink = 0;
    const size_t chunk = 64 << 20;
    const double per_chunk_s = (double) (chunk / (1024 * 1024)) / (rate_mbs > 0 ? rate_mbs : 1000.0);
    auto next = std::chrono::steady_clock::now();
    fprintf(stderr, "memhog: streaming %zu GiB at %.0f MB/s\n", gb, rate_mbs);
    fflush(stderr);
    for (;;) {
        for (size_t off = 0; off + chunk <= n; off += chunk) {
            for (size_t i = 0; i < chunk; i += 64) {
                sink += (unsigned char) p[off + i];
            }
        }
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(per_chunk_s));
        std::this_thread::sleep_until(next);
    }
    return 0;
}
