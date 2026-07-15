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
    
    bool parse_gguf_manifest();
    bool parse_gguf_metadata();
    void parse_tensor_dims_from_file(
        std::unordered_map<std::string, std::array<int64_t, 4>>& out_dims,
        std::unordered_map<std::string, uint32_t>& out_n_dims);
    std::future<void> read_expert_async(const ExpertManifestEntry& entry,
                                         std::shared_ptr<ExpertTensorBuffer> target_buf);
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
GUANACO_API bool apply_madvise_dontneed(void* addr, size_t length);

} // namespace guanaco