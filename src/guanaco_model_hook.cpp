#include "guanaco/guanaco_model_hook.h"
#include "guanaco/guanaco.h"

#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstring>

namespace guanaco {

// Concrete GuanacoModelHook implementation used by llama.cpp.
// Iteration 2: Router Prefetch + Tensor-Data Redirection.
//  - register_expert_tensor(): a fused expert tensor (e.g. blk.N.ffn_up_exps.weight)
//    gets a sparse anonymous-mmap slab mirroring the fused layout.
//  - get_expert_tensor_data(): llama.cpp points the ggml tensor's ->data at this slab.
//  - on_router_computed(): after the ffn_moe_topk node is computed, the
//    router-selected expert ids are used to pread ONLY those expert slices
//    from the GGUF into the slabs (so mul_mat_id reads them resident).
struct GuanacoModelHookImpl : public GuanacoModelHook {
    SteppeLoaderConfig sl_config;
    std::unique_ptr<SteppeLoader> loader;
    std::string model_path;

    GuanacoModelHookImpl(const char* path, size_t max_active_experts)
        : model_path(path ? path : "") {
        sl_config.gguf_path      = model_path;
        sl_config.max_active_experts = (int)max_active_experts;
        sl_config.use_madvise   = true;
        sl_config.use_io_uring  = true;
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

    const ExpertManifestEntry* lookup_expert(const char* tensor_name) const override {
        if (!loader || !tensor_name) return nullptr;
        return loader->lookup_expert(tensor_name);
    }

    void register_expert_tensor(const char* tensor_name, int layer_idx,
                                 size_t file_offset, size_t byte_size, int num_experts) override {
        if (!loader || !tensor_name) return;
        loader->register_expert_tensor(tensor_name, layer_idx, file_offset, byte_size, num_experts);
    }

    void* get_expert_tensor_data(const char* tensor_name) override {
        if (!loader || !tensor_name) return nullptr;
        return loader->get_expert_tensor_data(tensor_name);
    }

    void on_router_computed(int layer_idx, const int* expert_ids, int n) override {
        if (!loader || expert_ids == nullptr || n <= 0) return;
        loader->prefetch_experts(layer_idx, expert_ids, n);
    }

    void advise_random_access(void* ml) override {
        (void)ml;
        if (loader) {
            loader->advise_experts_random();
        }
    }
};

void* create_guanaco_model_hook(const char* model_path, size_t max_active_experts) {
    return new GuanacoModelHookImpl(model_path, max_active_experts);
}

void destroy_guanaco_model_hook(void* hook) {
    delete static_cast<GuanacoModelHookImpl*>(hook);
}

} // namespace guanaco
