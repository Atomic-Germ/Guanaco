#include "guanaco/guanaco_model_hook.h"
#include "guanaco/guanaco.h"

#include <iostream>
#include <vector>
#include <cstring>

struct GuanacoModelHookImpl {
    guanaco::GuanacoExecutorConfig config;
    std::unique_ptr<guanaco::GuanacoExecutor> executor;
    
    GuanacoModelHookImpl(const char* model_path, size_t max_active_experts) 
        : config() {
        config.model_path = model_path ? model_path : "";
        config.max_active_experts = (int)max_active_experts;
        config.use_madvise = true;
        config.use_io_uring = false;
        config.num_layers = 32;
        config.num_experts = 8;
        config.top_k = 2;
        executor = std::make_unique<guanaco::GuanacoExecutor>(config);
    }
    
    ~GuanacoModelHookImpl() = default;
};

extern "C" {

void* create_guanaco_model_hook(const char* model_path, size_t max_active_experts) {
    return new GuanacoModelHookImpl(model_path, max_active_experts);
}

void destroy_guanaco_model_hook(void* hook) {
    delete static_cast<GuanacoModelHookImpl*>(hook);
}

}