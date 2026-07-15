#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <unordered_map>
#include <functional>

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
    void *      slab            = nullptr;
    std::vector<bool> loaded;              // per-expert residency bits
    std::vector<uint64_t> total_hits;      // cumulative router hits per expert
    std::vector<float>   window_hits;     // decaying windowed hits (EWMA)
    std::vector<bool>    pinned;          // hot experts kept resident
};

struct GUANACO_API SteppeLoaderConfig {
    std::string gguf_path;
    int max_active_experts = 2;
    bool use_io_uring = true;
    bool use_madvise = true;
    size_t io_queue_depth = 256;
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

    // Register a fused expert tensor for slab redirection. Allocates a
    // sparse anonymous-mmap slab mirroring the fused layout; only selected
    // expert slices are ever paged in (via prefetch_experts).
    void register_expert_tensor(const std::string& name, int layer,
                              size_t file_offset, size_t byte_size, int num_experts);

    // Pointer to the slab allocated for a registered expert tensor.
    void* get_expert_tensor_data(const std::string& name);

    // Look up a tensor's parsed GGUF manifest entry (for file offsets).
    const ExpertManifestEntry* lookup_expert(const std::string& name) const;

    // Record router decisions for `layer`. Accumulates per-expert hotness
    // (EWMA window) and, once warmed up, permanently pins the hottest
    // experts so their slab slices stay resident instead of being
    // re-streamed from disk on every occurrence.
    void record_routing(int layer, const int* expert_ids, int n);

    // Prefetch the given (router-selected) expert ids for every registered
    // expert tensor that belongs to `layer`. Synchronous pread into the
    // slabs at the correct fused-layout offsets.
    void prefetch_experts(int layer, const int* expert_ids, int n);

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
    MoEModelConfig model_config_;
    std::unordered_map<std::string, ExpertTensor> expert_tensors_;

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
    bool pread_full(off_t offset, void* dst, size_t len);

    // Hot-expert pinning state.
    static constexpr float   kHitDecay_   = 0.95f;   // EWMA decay per record_routing()
    static constexpr size_t  kWarmupLayers_   = 200;  // per-layer records before pinning
    static constexpr size_t  kLogEveryLayers_ = 400;  // summary cadence (per-layer records)
    size_t                   routing_records_ = 0;
    size_t                   pinned_total_    = 0;
    void maybe_pin_hot_experts();
    void log_hot_summary();
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
#endif
GUANACO_API bool apply_madvise_dontneed(void* addr, size_t length);

} // namespace guanaco