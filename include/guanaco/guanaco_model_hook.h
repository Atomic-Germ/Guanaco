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

// Forward declarations for llama.cpp types
struct llama_model_loader;
struct llama_model;
struct llama_layer;

#ifdef __cplusplus
namespace guanaco {

// Abstract interface for llama.cpp model hook
struct GUANACO_API GuanacoModelHook {
    virtual ~GuanacoModelHook() = default;
    
    // Initialize hook during model loading
    virtual void initialize(void* ml, void* model) = 0;
    
    // Check if expert tensor should be streamed
    virtual bool should_stream_expert(const char* tensor_name, int layer_idx, int expert_idx) = 0;
    
    // Register expert tensor with hook
    virtual void register_tensor(const char* tensor_name, int layer_idx, int expert_idx) = 0;
    
    // Register all expert tensors for a layer
    virtual void register_expert_tensors(void* layer, int layer_idx, void* ml) = 0;
    
    // Get buffer for expert tensor data
    virtual void* get_expert_buffer(int layer_idx, int expert_idx, size_t size) = 0;
    
    // Redirect tensor data pointer to streaming buffer
    virtual void redirect_tensor_data(void* tensor, int layer_idx, int expert_idx, int tensor_type) = 0;
    
    // Prefetch experts for a layer
    virtual void prefetch_experts(int layer_idx, const int* expert_ids, int num_experts) = 0;
    
    // Called when router computes logits
    virtual void on_router_computed(int layer_idx, const float* logits, int num_experts, int top_k, int num_tokens) = 0;
    
    // Release expert buffers after computation
    virtual void release_experts(int layer_idx, const int* expert_ids, int num_experts) = 0;
    
    // Apply MADV_RANDOM to expert regions
    virtual void advise_random_access(void* ml) = 0;
    
    // Skip loading expert tensor data
    virtual void skip_expert_tensor_load(void* ml) = 0;
};

// Factory function to create a model hook
GUANACO_API void* create_guanaco_model_hook(const char* model_path, size_t max_active_experts);
GUANACO_API void destroy_guanaco_model_hook(void* hook);

} // namespace guanaco
#endif