#ifdef GUANACO_HAVE_IO_URING
#include "guanaco/guanaco.h"
#include <liburing.h>
#include <vector>
#include <algorithm>

namespace guanaco {

// Batched io_uring reader for slab prefetch. One ring instance is created
// per SteppeLoader; a layer's slices are submitted in one batch and reaped
// together. This lets the kernel dispatch the NVMe reads in parallel and
// avoids N separate pread() syscalls.
struct IoUringSlab {
    struct io_uring ring;
    bool ok = false;
    explicit IoUringSlab(unsigned entries) {
        ok = (io_uring_queue_init(entries, &ring, 0) == 0);
    }
    ~IoUringSlab() {
        if (ok) io_uring_queue_exit(&ring);
    }
};

void* io_uring_slab_create(unsigned entries) {
    auto* r = new IoUringSlab(entries);
    if (!r->ok) {
        delete r;
        return nullptr;
    }
    return r;
}

void io_uring_slab_destroy(void* ctx) {
    delete static_cast<IoUringSlab*>(ctx);
}

// Queue every slice in `slices` and submit, tagging each completion with
// `token`. Returns the number of requests queued, or -1 on failure. Does not
// wait for the kernel; call io_uring_slab_reap to collect completions. If the
// submission queue fills mid-batch it is drained and re-queued, so the batch
// should be smaller than the CQ depth (dense layers use about a dozen slices).
int io_uring_slab_submit(void* ctx, int fd, const std::vector<IoUringSlice>& slices, int64_t token) {
    auto* r = static_cast<IoUringSlab*>(ctx);
    if (!r || !r->ok || fd < 0) {
        return -1;
    }
    int queued = 0;
    int pending = 0;
    for (const auto& s : slices) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&r->ring);
        if (!sqe) {
            if (io_uring_submit(&r->ring) < 0) return -1;
            pending = 0;
            sqe = io_uring_get_sqe(&r->ring);
            if (!sqe) break; // queue genuinely stuck
        }
        io_uring_prep_read(sqe, fd, s.dst, static_cast<unsigned>(s.len), static_cast<uint64_t>(s.offset));
        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(static_cast<intptr_t>(token)));
        queued++;
        pending++;
    }
    if (pending > 0 && io_uring_submit(&r->ring) < 0) {
        return -1;
    }
    return queued > 0 ? queued : -1;
}

// Wait for one completion and report its token and result. Returns false only
// on a hard io_uring error.
bool io_uring_slab_reap(void* ctx, int64_t* token, int* res) {
    auto* r = static_cast<IoUringSlab*>(ctx);
    if (!r || !r->ok) {
        return false;
    }
    struct io_uring_cqe* cqe = nullptr;
    if (io_uring_wait_cqe(&r->ring, &cqe) < 0) {
        return false;
    }
    if (token) {
        *token = static_cast<int64_t>(reinterpret_cast<intptr_t>(io_uring_cqe_get_data(cqe)));
    }
    if (res) {
        *res = cqe->res;
    }
    io_uring_cqe_seen(&r->ring, cqe);
    return true;
}

bool io_uring_slab_read(void* ctx, int fd, const std::vector<IoUringSlice>& slices) {
    auto* r = static_cast<IoUringSlab*>(ctx);
    if (!r || !r->ok || fd < 0 || slices.empty()) {
        return false;
    }
    bool all_ok = true;
    size_t start = 0;
    const size_t chunk = 128;  // well under the queue depth
    while (start < slices.size()) {
        const size_t end = std::min(start + chunk, slices.size());
        std::vector<IoUringSlice> part(slices.begin() + start, slices.begin() + end);
        const int n = io_uring_slab_submit(ctx, fd, part, 0);
        if (n < 0) {
            all_ok = false;
            break;
        }
        for (int k = 0; k < n; ++k) {
            int64_t token = 0;
            int res = 0;
            if (!io_uring_slab_reap(ctx, &token, &res)) {
                all_ok = false;
                break;
            }
            if (res < 0) all_ok = false;
        }
        start = end;
    }
    return all_ok;
}

} // namespace guanaco
#endif
