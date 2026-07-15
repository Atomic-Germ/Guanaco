#include "guanaco/guanaco_model_hook.h"
#include "guanaco/guanaco.h"

#include <iostream>
#include <vector>
#include <cstring>

namespace guanaco {

// Concrete GuanacoModelHook implementation used by llama.cpp.
// For Iteration 1 it parses the GGUF expert manifest via SteppeLoader and
// applies MADV_RANDOM to expert tensor regions so the OS kernel does not
// thrash with sequential read-ahead when experts are accessed randomly.
struct GuanacoModelHookImpl : public GuanacoModelHook {
    SteppeLoaderConfig sl_config;
    std::unique_ptr<SteppeLoader> loader;
    std::vector<ExpertManifestEntry> registered;
    std::string model_path;

    GuanacoModelHookImpl(const char* path, size_t max_active_experts)
        : model_path(path ? path : "") {
        sl_config.gguf_path      = model_path;
        sl_config.max_active_experts = (int)max_active_experts;
        sl_config.use_madvise   = true;
        sl_config.use_io_uring  = false;
        loader = std::make_unique<SteppeLoader>(sl_config);
    }

    ~GuanacoModelHookImpl() override = default;

    void initialize(void* ml, void* model) override {
        (void)ml;
        (void)model;
        if (loader) {
            loader->initialize();
        }
    }

    bool should_stream_expert(const char* tensor_name, int layer_idx, int expert_idx) override {
        (void)tensor_name;
        (void)layer_idx;
        (void)expert_idx;
        return true; // Iteration 1: mark all expert tensors for streaming
    }

    void register_tensor(const char* tensor_name, int layer_idx, int expert_idx) override {
        ExpertManifestEntry e;
        e.tensor_name = tensor_name ? tensor_name : "";
        e.layer_idx   = layer_idx;
        e.expert_idx  = expert_idx;
        registered.push_back(std::move(e));
    }

    void register_expert_tensors(void* layer, int layer_idx, void* ml) override {
        (void)layer;
        (void)layer_idx;
        (void)ml;
    }

    void* get_expert_buffer(int layer_idx, int expert_idx, size_t size) override {
        (void)layer_idx;
        (void)expert_idx;
        (void)size;
        return nullptr;
    }

    void redirect_tensor_data(void* tensor, int layer_idx, int expert_idx, int tensor_type) override {
        (void)tensor;
        (void)layer_idx;
        (void)expert_idx;
        (void)tensor_type;
    }

    void prefetch_experts(int layer_idx, const int* expert_ids, int num_experts) override {
        (void)layer_idx;
        (void)expert_ids;
        (void)num_experts;
    }

    void on_router_computed(int layer_idx, const float* logits, int num_experts, int top_k, int num_tokens) override {
        (void)layer_idx;
        (void)logits;
        (void)num_experts;
        (void)top_k;
        (void)num_tokens;
    }

    void release_experts(int layer_idx, const int* expert_ids, int num_experts) override {
        (void)layer_idx;
        (void)expert_ids;
        (void)num_experts;
    }

    void advise_random_access(void* ml) override {
        (void)ml;
        if (loader) {
            loader->advise_experts_random();
        }
    }

    void skip_expert_tensor_load(void* ml) override {
        (void)ml;
    }
};

void* create_guanaco_model_hook(const char* model_path, size_t max_active_experts) {
    return new GuanacoModelHookImpl(model_path, max_active_experts);
}

void destroy_guanaco_model_hook(void* hook) {
    delete static_cast<GuanacoModelHookImpl*>(hook);
}

} // namespace guanaco
