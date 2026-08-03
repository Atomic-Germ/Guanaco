#ifdef GUANACO_HAVE_IO_URING
#include "guanaco/guanaco.h"
#include <liburing.h>
#include <vector>
#include <algorithm>

namespace guanaco {

// Batched io_uring reader for slab prefetch. One ring instance is created
// per SteppeLoader; prefetch_experts() submits every selected expert slice
// for a layer in a single batch and reaps the completions together. This
// lets the kernel dispatch the NVMe reads in parallel and avoids N separate
// pread() syscalls.
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

bool io_uring_slab_read(void* ctx, int fd, const std::vector<IoUringSlice>& slices) {
    auto* r = static_cast<IoUringSlab*>(ctx);
    if (!r || !r->ok || fd < 0 || slices.empty()) {
        return false;
    }

    bool all_ok = true;
    size_t start = 0;
    const size_t chunk = 128;  // well under the queue depth
    while (start < slices.size()) {
        size_t end = std::min(start + chunk, slices.size());
        unsigned inflight = 0;
        for (size_t i = start; i < end; ++i) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&r->ring);
            if (!sqe) break;  // SQ full: submit what we have, continue
            io_uring_prep_read(sqe, fd,
                               slices[i].dst,
                               static_cast<unsigned>(slices[i].len),
                               static_cast<uint64_t>(slices[i].offset));
            io_uring_sqe_set_data(sqe, nullptr);
            inflight++;
        }
        if (inflight == 0) { all_ok = false; break; }
        if (io_uring_submit(&r->ring) < 0) { all_ok = false; break; }
        for (unsigned k = 0; k < inflight; ++k) {
            struct io_uring_cqe* cqe = nullptr;
            if (io_uring_wait_cqe(&r->ring, &cqe) < 0) { all_ok = false; break; }
            if (cqe->res < 0) all_ok = false;
            io_uring_cqe_seen(&r->ring, cqe);
        }
        start = end;
    }
    return all_ok;
}

} // namespace guanaco
#endif
