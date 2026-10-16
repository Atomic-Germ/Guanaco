#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <unordered_map>
#include <functional>

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

#ifdef _WIN32
    #ifdef GUANACO_EXPORTS
        #define GUANACO_API __declspec(dllexport)
    #else
        #define GUANACO_API __declspec(dllimport)
    #endif
#else
    #define GUANACO_API __attribute__((visibility("default")))
#endif

namespace guanaco {

struct GUANACO_API ExpertManifestEntry {
    std::string tensor_name;
    size_t file_offset;
    size_t byte_size;
    int layer_idx;
    int expert_idx;
    int tensor_type;
    std::string quantization_type;
    int num_experts_in_tensor = 1;
    // Expert slice info (for fused tensors)
    size_t expert_slice_offset = 0;
    size_t expert_slice_size = 0;
    int64_t tensor_dims[4] = {1, 1, 1, 1};
    uint32_t tensor_n_dims = 1;
};

struct GUANACO_API ExpertTensorBuffer {
    std::vector<char> data;
    bool is_ready = false;
    int expert_id = -1;
    int layer_idx = -1;
};

// A fused expert tensor (e.g. blk.N.ffn_up_exps.weight) mirrored into a
// sparse anonymous-mmap slab. Only the expert slices that the router
// selects are ever paged in (via pread); the rest stay demand-zero.
struct GUANACO_API ExpertTensor {
    std::string name;
    int         layer            = -1;
    size_t      file_offset     = 0;   // absolute offset in the GGUF file
    size_t      byte_size       = 0;   // total fused tensor size in bytes
    int         num_experts     = 1;   // expert dimension of the fused tensor
    size_t      per_expert_bytes = 0;   // byte_size / num_experts
    // The original mmap base of the fused tensor, captured at registration.
    // ggml reads the expert weights from this address; we never redirect it.
    // Disk streaming is done in-place by (de)pressing these mmap pages with
    // madvise(MADV_DONTNEED / MADV_WILLNEED) so residency is controlled at
    // the SAME address ggml consumes.
    void *      mmap_base       = nullptr;
    std::vector<bool> loaded;              // per-expert residency bits
    std::vector<uint64_t> total_hits;      // cumulative router hits per expert
    std::vector<float>   window_hits;     // decaying windowed hits (EWMA)
    std::vector<bool>    pinned;          // hot experts kept resident
    std::vector<uint64_t> last_used;       // LRU recency clock per expert slice
    // Per-(from->to) routing transition counts, learned live. Drives the
    // cross-layer PILOT prefetch: given the experts selected at this layer,
    // the most likely experts at the next layer are read ahead while the
    // current MoE block computes. Row-major over num_experts^2.
    std::vector<uint64_t> trans;          // num_experts * num_experts
    // Per-tensor dynamic resident budget (slots). Rebalanced by
    // SteppeLoader::recompute_budgets() so busy tensors get more slots than
    // idle ones, while the aggregate footprint stays within the global
    // max_active_experts * n_tensors envelope. Initialized to the global
    // budget at registration; never exceeds it.
    int         budget          = 0;
};

struct GUANACO_API SteppeLoaderConfig {
    std::string gguf_path;
    int max_active_experts = 2;
    bool use_io_uring = true;
    bool use_madvise = true;
    bool use_pilot = true;     // cross-layer lookahead prefetch (default on)
    // Pilot prefetch pruning: cover at most this fraction of the cumulative
    // (from->to) transition mass, keeping only the dominant predicted experts
    // and dropping the long tail. 1.0 = read the full top-K (no pruning).
    float pilot_mass = 0.9f;
    bool use_imatrix = true;   // seed hot pins from "<model>.imatrix.gguf" prior
    bool dontneed = true;      // madvise(DONTNEED) slab slices after copy (drop page-cache)
    size_t io_queue_depth = 256;

    // Dense layer-window streaming. For a dense model (no MoE experts) there is
    // no routing, so instead of a per-expert cache we keep a sliding window of
    // transformer layers resident and page the rest in from disk. Only active
    // when dense_budget_bytes > 0; otherwise dense models keep the plain mmap.
    bool   dense_streaming    = false;
    size_t dense_budget_bytes = 0;   // RAM budget for the resident layer window
    int    dense_prefetch_ahead = 3; // layers to WILLNEED ahead of the current one
    // Read dense tensors into private anonymous slabs (DONTNEED actually frees
    // them). When false, fall back to in-place madvise on the shared mmap.
    bool   dense_use_slab = true;
};

// One whole dense weight tensor registered for layer-window streaming. Unlike
// ExpertTensor there is no routing: the whole tensor is read into a private
// anonymous slab and residency is controlled per layer by DONTNEED/WILLNEED
// (DONTNEED actually frees the slab pages, unlike on the shared file mmap).
// ggml reads the tensor via a redirected t->data pointer into `slab`.
struct GUANACO_API StreamTensor {
    std::string name;
    int    layer       = -1;
    size_t file_offset = 0;
    size_t byte_size   = 0;
    void * mmap_base   = nullptr;   // original file-backed mmap address (fallback)
    void * slab        = nullptr;   // private anon slab the tensor is read into
    size_t slab_len    = 0;         // page-rounded slab length in bytes
};

struct GUANACO_API MoEModelConfig {
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int num_hidden_layers = 0;
    int hidden_size = 0;
    std::string architecture;
};

class GUANACO_API SteppeLoader {
public:
    explicit SteppeLoader(const SteppeLoaderConfig& config);
    ~SteppeLoader();

    bool initialize();
    void shutdown();

    // Apply MADV_RANDOM to expert tensor regions so the OS kernel does not
    // perform sequential read-ahead when experts are accessed randomly.
    void advise_experts_random();

    // Register a fused expert tensor for in-place mmap paging. Records the
    // tensor's mmap base address (which ggml reads from directly) and the
    // per-expert slice geometry; residency is then controlled via madvise
    // without ever redirecting t->data.
    void register_expert_tensor(const std::string& name, int layer,
                              size_t file_offset, size_t byte_size, int num_experts,
                              void* mmap_base = nullptr);

    // Dense layer-window streaming (no MoE routing). Register a whole dense
    // weight tensor; on_layer_boundary() pages whole layers in and out with
    // madvise so only dense_budget_bytes worth of layers stay resident.
    void register_stream_tensor(const std::string& name, int layer,
                              size_t file_offset, size_t byte_size, void* mmap_base);
    void on_layer_boundary(int layer);
    bool dense_streaming() const { return dense_enabled_; }

    // Pointer to the slab allocated for a registered dense stream tensor, or
    // nullptr if the tensor was not registered / has no slab. llama.cpp points
    // the ggml tensor's ->data at this address after registration.
    void* get_stream_tensor_data(const std::string& name);

    // Read the fixed resident subset of layers ([0, W)) into their slabs.
    // Called once after all dense tensors are registered so the first token
    // finds its weights resident.
    void fill_resident_layers();

    // Pointer to the mmap base recorded for a registered expert tensor.
    void*  get_expert_tensor_data(const std::string& name);

    // Look up a tensor's parsed GGUF manifest entry (for file offsets).
    const ExpertManifestEntry* lookup_expert(const std::string& name) const;

    // Seed per-expert hotness from a sibling "<model>.imatrix.gguf" prior.
    // The imatrix file stores, per expert tensor, a `counts` vector of how
    // often each expert was selected during calibration. Injecting those
    // totals into total_hits[] gives hot-expert pinning a correct head start
    // from token 0, before any runtime routing statistics exist. If no
    // sibling imatrix exists the loader silently falls back to runtime-only
    // pinning (the existing behavior). Returns the number of experts whose
    // totals were seeded. Subsequent calls to on_expert_tensors_ready()
    // perform the initial pin pass.
    size_t load_imatrix_prior(const std::string& model_path);

    // Record router decisions for `layer`. Accumulates per-expert hotness
    // (EWMA window) and, once warmed up, permanently pins the hottest
    // experts so their slab slices stay resident instead of being
    // re-streamed from disk on every occurrence.
    void record_routing(int layer, const int* expert_ids, int n);

    // Prefetch the given (router-selected) expert ids for every registered
    // expert tensor that belongs to `layer`. Synchronous pread into the
    // slabs at the correct fused-layout offsets.
    void prefetch_experts(int layer, const int* expert_ids, int n);

    // Cross-layer PILOT prefetch: given the experts selected at `layer`,
    // predict the most likely experts at `layer+1` from the learned
    // per-layer transition matrix and stream them into their slabs ahead
    // of the real router decision. Hint-only: a wrong prediction just
    // wastes a disk read into a cold slab; it never evicts a hot expert.
    void pilot_prefetch_next_layer(int layer, const int* expert_ids, int n);

    // Snapshot of the hottest experts per layer for diagnostics/logging.
    struct HotExpertStats {
        int      layer       = -1;
        int      expert      = -1;
        uint64_t total_hits  = 0;
        float    window_hits = 0.0f;
        bool     pinned      = false;
    };
    std::vector<HotExpertStats> get_hot_expert_stats() const;

    const std::vector<ExpertManifestEntry>& get_manifest() const { return manifest_; }
    const MoEModelConfig& get_model_config() const { return model_config_; }
    
    std::future<void> prefetch_expert(int layer_idx, int expert_idx, 
                                       std::shared_ptr<ExpertTensorBuffer> target_buf);
    
    std::future<void> prefetch_experts(int layer_idx, const std::vector<int>& expert_indices,
                                         const std::vector<std::shared_ptr<ExpertTensorBuffer>>& target_bufs);

    int get_fd() const { return fd_; }

private:
    SteppeLoaderConfig config_;
    int fd_ = -1;
    std::vector<ExpertManifestEntry> manifest_;
    std::unordered_map<int, std::vector<ExpertManifestEntry>> layer_experts_;
    std::vector<int>          moe_layers_;      // sorted unique MoE layer indices
    MoEModelConfig model_config_;
    std::unordered_map<std::string, ExpertTensor> expert_tensors_;
    // When false the model has no MoE experts (dense model, or the expert
    // count could not be determined): streaming is disabled and every method
    // below becomes a no-op, so ggml uses its normal mmap and the model just
    // runs. This prevents mis-parsing a dense FFN tensor as a 1-"expert" block.
    bool enabled_ = true;

    // Dense layer-window streaming state. Populated only when the model has no
    // MoE experts and dense_budget_bytes > 0.
    bool dense_enabled_ = false;
    std::unordered_map<int, std::vector<StreamTensor>> stream_layers_; // whole tensors per layer
    std::vector<char> layer_resident_;      // per-layer residency bookkeeping (window)
    size_t stream_total_bytes_ = 0;         // sum of registered dense tensor bytes
    size_t dense_budget_bytes_ = 0;         // resident layer-window budget in bytes
    int    stream_n_layers_    = 0;         // highest registered layer + 1
    int    last_boundary_layer_ = -1;       // last l_out layer seen (new-pass detection)
    int    dense_boundaries_   = 0;         // on_layer_boundary calls that did work
    long   dense_prefetch_ops_ = 0;         // layers read into slabs
    long   dense_evict_ops_    = 0;         // layers DONTNEED'd back out
    long   dense_lock_fails_   = 0;         // slab allocation failures
    long   dense_uring_ops_    = 0;         // layers read via io_uring
    long   dense_pread_ops_    = 0;         // layers read via pread fallback
    double dense_io_seconds_   = 0.0;       // cumulative time in slab reads
    int    stream_resident_layers_ = 0;     // fixed resident subset size W (layers [0,W))
    long   dense_wait_ops_     = 0;         // ensure_layer calls that had to wait
    int    submitted_upto_     = -1;        // highest streamed layer queued to io_uring
    uint64_t madvise_ops_      = 0;         // successful madvise_range calls
    uint64_t madvise_failures_ = 0;         // rejected by the kernel (e.g. EINVAL)
    // Global expert residency, tracked so the expert path can honour a byte
    // budget rather than only the per-tensor expert-count cap.
    uint64_t expert_resident_bytes_ = 0;
    size_t   expert_budget_bytes_   = 0;   // 0 = no ceiling (no -gdb given)
    size_t   stream_resident_bytes_ = 0;   // bytes held by the layer pager

    void prefetch_layer(int layer);
    void evict_layer(int layer);
    void* alloc_slab(size_t byte_size);
    // Page-aligns and issues madvise(), counting successes and failures so a
    // silently rejected advice call cannot go unnoticed.
    int  madvise_range(void* base, size_t size, int advice);
    // Submit a layer's slab reads to io_uring without waiting. Returns false if
    // io_uring is unavailable or the submit failed (caller falls back to sync).
    bool submit_layer_async(int layer);
    // Make sure a streamed layer is resident before it is computed, reaping its
    // in-flight io_uring reads if needed. Synchronous only on a cold pipeline.
    void ensure_layer(int layer);
    void drain_inflight();
    std::unordered_map<int, int> dense_inflight_; // layer -> outstanding completions

#ifdef GUANACO_HAVE_IO_URING
    void* uring_slab_ = nullptr;  // batched io_uring reader for slab prefetch
#endif

    bool parse_gguf_manifest();
    bool parse_gguf_metadata();
    void parse_tensor_dims_from_file(
        std::unordered_map<std::string, std::array<int64_t, 4>>& out_dims,
        std::unordered_map<std::string, uint32_t>& out_n_dims);
    std::future<void> read_expert_async(const ExpertManifestEntry& entry,
                                         std::shared_ptr<ExpertTensorBuffer> target_buf);

    // Hot-expert pinning state.
    static constexpr float   kHitDecay_   = 0.95f;   // EWMA decay per record_routing()
    static constexpr float   kHotProtect_ = 1.0f;    // window_hits above this are protected from PASS-A eviction
    static constexpr size_t  kWarmupLayers_   = 200;  // per-layer records before pinning
    static constexpr size_t  kLogEveryLayers_ = 400;  // summary cadence (per-layer records)
    static constexpr size_t  kPinRateEvery_   = 50;   // pin-cache hit-rate log cadence (calls)
    size_t                   routing_records_ = 0;
    size_t                   pinned_total_    = 0;   // cumulative pin events
    size_t                   pinned_current_ = 0;   // distinct experts pinned now
    bool                    imatrix_seeded_  = false;  // load attempted (don't retry)
    bool                    imatrix_loaded_ = false;  // prior actually applied to pins
    // Pin-cache hit-rate instrumentation: counts of router-selected expert
    // accesses that were served by an already-pinned (hot) or already-loaded
    // (warm) slice versus ones that required a fresh disk read.
    uint64_t                 pin_hits_   = 0;  // served by a pinned expert
    uint64_t                 warm_hits_  = 0;  // served by an already-loaded (non-pinned) slice
    uint64_t                 disk_miss_  = 0;  // required a disk read
    uint64_t                 prefetch_calls_ = 0;
    uint64_t                 pilot_calls_ = 0;     // pilot prefetch invocations
    uint64_t                 pilot_slice_reads_ = 0; // expert slices pulled by pilot
    uint64_t                 evictions_ = 0;       // LRU slab slices reclaimed
    // Eviction-churn instrumentation (tuning data only; no behavior change):
    //  - evict_hot_:   evictions of experts with meaningful windowed hotness
    //                   (these are the candidates a hot-protect policy would save)
    //  - evict_reread_: evictions where the same (tensor,id) was later read
    //                   back from disk within kChurnWindow eclock ticks (wasted)
    uint64_t                 evict_hot_      = 0;
    uint64_t                 evict_reread_  = 0;
    struct EvictRec { ExpertTensor* t; int id; uint64_t clock; bool used; };
    static constexpr uint64_t kChurnWindow_ = 64;   // eclock ticks (~tokens) to count a re-read as churn
    std::vector<EvictRec>    evict_log_;             // bounded FIFO of recent evictions
    static constexpr size_t   kEvictLogMax_ = 8192;
    int                      pilot_from_layer_ = -1;  // layer whose ids are pending as "from"
    std::vector<int>         pilot_from_ids_;        // selected ids of pilot_from_layer_
    uint64_t                 eclock_ = 0;            // monotonic LRU recency counter
    void maybe_pin_hot_experts();
    // Rebalance per-tensor resident budgets (t.budget) from observed distinct-
    // expert usage so the global max_active_experts cap is allocated where it
    // is actually needed. Idle tensors shrink, busy ones grow; the aggregate
    // stays within max_active_experts * n_tensors.
    void recompute_budgets();
    // Evict globally least-recently-used non-pinned expert slices across every
    // tensor until expert residency is back under expert_budget_bytes_.
    void enforce_expert_byte_ceiling();
    // LRU eviction: keep at most max_active_experts resident (pinned +
    // warm) slices per tensor. Non-pinned slices are evicted least-
    // recently-used first; their slab pages are dropped (safe because a
    // cleared loaded[] flag forces a re-read before ggml uses the slice).
    void evict_to_budget(ExpertTensor& t);
    void touch(ExpertTensor& t, int id);
    void pin_from_seeded_totals();   // initial pin pass driven by imatrix prior
    void log_hot_summary();
    void log_pin_rate();
    void log_final_summary();        // one-line end-of-run report
};

struct GUANACO_API HerdCacheConfig {
    size_t max_active_experts = 2;
    size_t max_buffer_size = 0;
    bool use_madvise = true;
};

class GUANACO_API HerdCache {
public:
    explicit HerdCache(const HerdCacheConfig& config);
    ~HerdCache();

    std::shared_ptr<ExpertTensorBuffer> allocate_slot(int layer_idx, int expert_idx);
    std::shared_ptr<ExpertTensorBuffer> get_buffer(int layer_idx, int expert_idx);
    
    bool is_pinned(int layer_idx, int expert_idx) const;
    void pin_expert(int layer_idx, int expert_idx);
    void unpin_expert(int layer_idx, int expert_idx);
    
    void evict_if_needed();
    size_t active_count() const { return active_experts_.size(); }
    void set_max_active(size_t max) { config_.max_active_experts = max; }

private:
    HerdCacheConfig config_;
    
    struct ExpertKey {
        int layer_idx;
        int expert_idx;
        bool operator==(const ExpertKey& other) const {
            return layer_idx == other.layer_idx && expert_idx == other.expert_idx;
        }
    };
    
    struct ExpertKeyHash {
        size_t operator()(const ExpertKey& k) const noexcept {
            return (static_cast<size_t>(k.layer_idx) << 32) ^ static_cast<size_t>(k.expert_idx);
        }
    };
    
    std::unordered_map<ExpertKey, std::shared_ptr<ExpertTensorBuffer>, ExpertKeyHash> active_experts_;
    std::vector<ExpertKey> lru_list_;
    
    void update_lru(const ExpertKey& key);
    void evict_lru();
    void apply_madvise(void* ptr, size_t size, bool will_need);
};

struct GUANACO_API RouterOutput {
    std::vector<int> selected_experts;
    std::vector<float> expert_weights;
    int layer_idx;
    int num_tokens = 0;
    int top_k = 0;
};

using RouterCallback = std::function<void(const RouterOutput&)>;

struct GUANACO_API GuanacoRouterConfig {
    int num_layers;
    int num_experts;
    int top_k;
    RouterCallback on_router_decision;
};

class GUANACO_API GuanacoRouter {
public:
    explicit GuanacoRouter(const GuanacoRouterConfig& config);
    ~GuanacoRouter();

    void intercept_router(int layer_idx, const float* router_logits, int num_tokens, int num_experts);
    void set_callback(RouterCallback callback) { config_.on_router_decision = callback; }

    void set_manifest(const std::vector<ExpertManifestEntry>* manifest);
    void set_cache(HerdCache* cache);
    void set_loader(SteppeLoader* loader);

    const std::vector<int>& get_selected_experts() const;
    const std::vector<float>& get_expert_weights() const;

private:
    GuanacoRouterConfig config_;
    std::vector<RouterOutput> pending_outputs_;
    const std::vector<ExpertManifestEntry>* manifest_ = nullptr;
    HerdCache* cache_ = nullptr;
    SteppeLoader* loader_ = nullptr;
    
    void process_router_output(int layer_idx, const float* logits, int num_tokens, int num_experts);
    void prefetch_experts(int layer_idx, const std::vector<int>& expert_ids);
    std::vector<int> top_k_indices(const float* logits, int num_experts, int top_k) const;
};

struct GUANACO_API GuanacoExecutorConfig {
    std::string model_path;
    int max_active_experts = 2;
    bool use_io_uring = true;
    bool use_madvise = true;
    int num_layers = 32;
    int num_experts = 8;
    int top_k = 2;
};

class GUANACO_API GuanacoExecutor {
public:
    explicit GuanacoExecutor(const GuanacoExecutorConfig& config);
    ~GuanacoExecutor();

    bool initialize();
    void evaluate_token(const std::vector<float>& hidden_states);
    void set_router_callback(std::function<void(int, const std::vector<int>&)> callback);

    SteppeLoader* loader() const { return loader_.get(); }
    HerdCache* cache() const { return cache_.get(); }
    GuanacoRouter* router() const { return router_.get(); }

private:
    GuanacoExecutorConfig config_;
    std::unique_ptr<SteppeLoader> loader_;
    std::unique_ptr<HerdCache> cache_;
    std::unique_ptr<GuanacoRouter> router_;
    std::vector<ExpertManifestEntry> manifest_;
    bool initialized_ = false;
    std::function<void(int, const std::vector<int>&)> router_callback_;
    
    void evaluate_moe_layer(int layer_idx, const std::vector<float>& token_hidden_states);
    void on_router_output(const RouterOutput& output);
};

GUANACO_API bool apply_madvise(void* addr, size_t length, int advice);
GUANACO_API bool apply_madvise_random(void* addr, size_t length);
GUANACO_API bool apply_madvise_willneed(void* addr, size_t length);

// Batched async io_uring read interface used by slab prefetch.
// Each slice is one contiguous expert region; all slices are submitted to the
// kernel in a single io_uring batch and reaped together.
#ifdef GUANACO_HAVE_IO_URING
struct IoUringSlice { void* dst; size_t len; int64_t offset; };
void* io_uring_slab_create(unsigned entries);
void  io_uring_slab_destroy(void* ctx);
 bool  io_uring_slab_read(void* ctx, int fd, const std::vector<IoUringSlice>& slices);
// Async split: queue the slices tagged with `token` and return the number
// queued (or -1 on error) without waiting; reap one completion at a time with
// io_uring_slab_reap. Lets the caller overlap reads with compute.
int  io_uring_slab_submit(void* ctx, int fd, const std::vector<IoUringSlice>& slices, int64_t token);
bool io_uring_slab_reap(void* ctx, int64_t* token, int* res);
#endif

} // namespace guanaco