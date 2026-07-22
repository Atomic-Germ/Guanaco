#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#ifdef _WIN32
    #ifdef GUANACO_EXPORTS
        #define GUANACO_API __declspec(dllexport)
    #else
        #define GUANACO_API __declspec(dllimport)
    #endif
#else
    #define GUANACO_API __attribute__((visibility("default")))
#endif

#include "guanaco/guanaco.h"

// Forward declarations for llama.cpp types
struct llama_model_loader;
struct llama_model;
struct llama_layer;

#ifdef __cplusplus
namespace guanaco {

// Abstract interface for llama.cpp model hook
struct GUANACO_API GuanacoModelHook {
    virtual ~GuanacoModelHook() = default;
    
    // Initialize hook during model loading (parses GGUF manifest)
    virtual void initialize(void* ml, void* model) = 0;
    
    // Look up a tensor's parsed GGUF manifest entry (file offset, etc.)
    virtual const ExpertManifestEntry* lookup_expert(const char* tensor_name) const = 0;
    
    // Register a fused expert tensor for slab redirection. Allocates a
    // sparse slab mirroring the fused layout; only selected expert
    // slices are ever paged in.
    virtual void register_expert_tensor(const char* tensor_name, int layer_idx,
                                    size_t file_offset, size_t byte_size, int num_experts,
                                    void* mmap_base = nullptr) = 0;
    
    // Pointer to the slab allocated for a registered expert tensor.
    virtual void* get_expert_tensor_data(const char* tensor_name) = 0;
    
    // Called when the router selects experts for a layer (post-compute of
    // the ffn_moe_topk node). Prefetches those experts into their slabs.
    virtual void on_router_computed(int layer_idx, const int* expert_ids, int n) = 0;

    // Called by llama.cpp once all expert tensors have been registered with
    // the SteppeLoader. Triggers imatrix-prior seeding and the initial
    // (cold-start) hot-expert pin pass so pinned experts are resident before
    // the first token is processed.
    virtual void on_expert_tensors_registered() = 0;

    // Streaming is done in-place on the file mmap: expert weights stay at
    // their original t->data address and residency is controlled via madvise,
    // so no tensor-data redirection is needed. This hook is now a no-op and
    // exists only to keep the load sequence symmetric. Returns empty.
    virtual std::vector<int> on_tensors_loaded() = 0;
    
    // Apply MADV_RANDOM to expert regions
    virtual void advise_random_access(void* ml) = 0;
};

// Factory function to create a model hook
// Accepts a fully populated SteppeLoaderConfig; all env-var / CLI
// processing is done upstream by llama.cpp/arg.cpp.
GUANACO_API void* create_guanaco_model_hook(const char* model_path, const SteppeLoaderConfig& config);
GUANACO_API void destroy_guanaco_model_hook(void* hook);

} // namespace guanaco
#endif