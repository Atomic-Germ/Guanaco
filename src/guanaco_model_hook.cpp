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
    // Tensor names registered via register_expert_tensor().  on_tensors_loaded()
    // pairs each name with the slab pointer so llama.cpp can redirect t->data
    // AFTER CPU_REPACK completes (avoiding the repack reading empty slab data).
    std::vector<std::string> registered_tensors_;

    GuanacoModelHookImpl(const char* path, size_t max_active_experts)
        : model_path(path ? path : "") {
        sl_config.gguf_path      = model_path;
        // -1 means "auto": keep a modest multiple of the router's top-k in
        // RAM so hot experts stay resident while cold ones stream on demand.
        // The GUANACO_MAX_EXPERTS env var overrides the API/default value,
        // letting users tune how many hot experts stay pinned per tensor.
        int resolved = (max_active_experts == size_t(-1)) ? 8 : (int)max_active_experts;
        const char* env_val = std::getenv("GUANACO_MAX_EXPERTS");
        if (env_val != nullptr) {
            int env_int = std::atoi(env_val);
            if (env_int > 0) {
                resolved = env_int;
            }
        }
        sl_config.max_active_experts = resolved;
        sl_config.use_madvise   = true;
        const char* iou_env = std::getenv("GUANACO_IO_URING");
        sl_config.use_io_uring  = (iou_env == nullptr) ? true : (std::atoi(iou_env) != 0);

        // Cross-layer PILOT lookahead prefetch: default ON (GUANACO_PILOT).
        // It only hints future reads and never evicts hot experts, so a
        // mispredict costs at most a wasted disk read, not correctness.
        const char* pilot_env = std::getenv("GUANACO_PILOT");
        sl_config.use_pilot = (pilot_env == nullptr) ? true : (std::atoi(pilot_env) != 0);

        // Pilot pruning: cover the top GUANACO_PILOT_MASS fraction of the
        // transition mass (default 0.9). 1.0 disables pruning (full top-K).
        const char* pmass_env = std::getenv("GUANACO_PILOT_MASS");
        if (pmass_env != nullptr) {
            float pm = std::atof(pmass_env);
            if (pm > 0.0f && pm <= 1.0f) sl_config.pilot_mass = pm;
        }

        // imatrix cold-start prior: default ON (GUANACO_IMATRIX).
        const char* imatrix_env = std::getenv("GUANACO_IMATRIX");
        sl_config.use_imatrix = (imatrix_env == nullptr) ? true : (std::atoi(imatrix_env) != 0);
        loader = std::make_unique<SteppeLoader>(sl_config);
        std::cerr << "[Guanaco HerdCache] pin budget (max_active_experts) = "
                  << resolved << " experts per tensor"
                  << (env_val ? " [from GUANACO_MAX_EXPERTS]" : "") << "\n";
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
                                 size_t file_offset, size_t byte_size, int num_experts,
                                 void* mmap_base = nullptr) override {
        if (!loader || !tensor_name) return;
        loader->register_expert_tensor(tensor_name, layer_idx, file_offset, byte_size, num_experts, mmap_base);
        registered_tensors_.emplace_back(tensor_name);
    }

    void* get_expert_tensor_data(const char* tensor_name) override {
        if (!loader || !tensor_name) return nullptr;
        return loader->get_expert_tensor_data(tensor_name);
    }

    void on_router_computed(int layer_idx, const int* expert_ids, int n) override {
        if (!loader || expert_ids == nullptr || n <= 0) return;
        loader->record_routing(layer_idx, expert_ids, n);
        loader->prefetch_experts(layer_idx, expert_ids, n);
        loader->pilot_prefetch_next_layer(layer_idx, expert_ids, n);
    }

    void advise_random_access(void* ml) override {
        (void)ml;
        if (loader) {
            loader->advise_experts_random();
        }
    }

    // Called once all expert tensors are registered: seed hot-expert
    // pinning from a sibling imatrix prior (if present), then pin the
    // calibration-hot experts so they are resident before token 0.
    void on_expert_tensors_registered() override {
        // Deferred to on_tensors_loaded() so CPU_REPACK completes first.
    }

    std::vector<int> on_tensors_loaded() override {
        // In-place mmap paging: no tensor-data redirection. Still seed the
        // hot-expert pin pass from a sibling imatrix prior here.
        if (loader) {
            loader->load_imatrix_prior(model_path);
        }
        return {};
    }
};

void* create_guanaco_model_hook(const char* model_path, size_t max_active_experts) {
    return new GuanacoModelHookImpl(model_path, max_active_experts);
}

void destroy_guanaco_model_hook(void* hook) {
    delete static_cast<GuanacoModelHookImpl*>(hook);
}

} // namespace guanaco
