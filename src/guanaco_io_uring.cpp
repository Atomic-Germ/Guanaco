#ifdef GUANACO_HAVE_IO_URING
#include "guanaco/guanaco.h"
#include <liburing.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace guanaco {

struct IoUringContext {
    struct io_uring ring;
    std::thread worker_thread;
    std::atomic<bool> running{true};
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<std::function<void()>> work_queue;
    
    IoUringContext(unsigned entries = 256) {
        io_uring_queue_init(entries, &ring, 0);
        worker_thread = std::thread([this]() { run_worker(); });
    }
    
    ~IoUringContext() {
        running = false;
        queue_cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        io_uring_queue_exit(&ring);
    }
    
    void run_worker() {
        while (running) {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return !work_queue.empty() || !running; });
                if (!running && work_queue.empty()) break;
                work = std::move(work_queue.front());
                work_queue.pop();
            }
            if (work) work();
        }
    }
    
    void submit_work(std::function<void()> work) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            work_queue.push(std::move(work));
        }
        queue_cv.notify_one();
    }
    
    std::future<void> read_async(int fd, void* buf, size_t count, off_t offset) {
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();
        
        submit_work([this, fd, buf, count, offset, promise]() {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                promise->set_exception(std::make_exception_ptr(std::runtime_error("io_uring full")));
                return;
            }
            
            struct iovec iov = { buf, count };
            io_uring_prep_readv(sqe, fd, &iov, 1, offset);
            io_uring_sqe_set_data(sqe, promise.get());
            
            int ret = io_uring_submit(&ring);
            if (ret < 0) {
                promise->set_exception(std::make_exception_ptr(std::runtime_error("io_uring_submit failed")));
                return;
            }
            
            struct io_uring_cqe* cqe;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                promise->set_exception(std::make_exception_ptr(std::runtime_error("io_uring_wait_cqe failed")));
                return;
            }
            
            if (cqe->res < 0) {
                promise->set_exception(std::make_exception_ptr(std::runtime_error("read failed")));
            } else {
                promise->set_value();
            }
            io_uring_cqe_seen(&ring, cqe);
        });
        
        return future;
    }
};

static thread_local std::unique_ptr<IoUringContext> g_io_uring_context;

IoUringContext& get_io_uring_context() {
    if (!g_io_uring_context) {
        g_io_uring_context = std::make_unique<IoUringContext>();
    }
    return *g_io_uring_context;
}

std::future<void> SteppeLoader::read_expert_io_uring(const ExpertManifestEntry& entry,
                                                      std::shared_ptr<ExpertTensorBuffer> target_buf) {
    auto& ctx = get_io_uring_context();
    target_buf->data.resize(entry.byte_size);
    target_buf->expert_id = entry.expert_idx;
    target_buf->layer_idx = entry.layer_idx;
    
    return ctx.read_async(fd_, target_buf->data.data(), entry.byte_size, entry.file_offset)
        .then([target_buf](std::future<void> fut) {
            try {
                fut.get();
                target_buf->is_ready = true;
            } catch (...) {
                target_buf->is_ready = false;
            }
        });
}

} // namespace guanaco
#endif