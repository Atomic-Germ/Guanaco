#include "guanaco/guanaco.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace guanaco {

GuanacoExecutor::GuanacoExecutor(const GuanacoExecutorConfig& config) : config_(config), initialized_(false) {
    SteppeLoaderConfig loader_config;
    loader_config.gguf_path = config_.model_path;
    loader_config.max_active_experts = config_.max_active_experts;
    loader_config.use_madvise = config_.use_madvise;
    
    loader_ = std::make_unique<SteppeLoader>(loader_config);
    
    HerdCacheConfig cache_config;
    cache_config.max_active_experts = config_.max_active_experts;
    cache_config.use_madvise = config_.use_madvise;
    cache_ = std::make_unique<HerdCache>(cache_config);
    
    router_ = std::make_unique<GuanacoRouter>(GuanacoRouterConfig{0, 0, 0, nullptr});
    
    router_->set_manifest(&manifest_);
    router_->set_cache(cache_.get());
    router_->set_loader(loader_.get());
    
    router_->set_callback([this](const RouterOutput& output) {
        on_router_output(output);
    });
}

GuanacoExecutor::~GuanacoExecutor() = default;

bool GuanacoExecutor::initialize() {
    if (!loader_->initialize()) {
        std::cerr << "[Guanaco] Failed to initialize loader" << std::endl;
        return false;
    }
    
    manifest_ = loader_->get_manifest();
    const MoEModelConfig& model_config = loader_->get_model_config();
    
    // Update router with actual model config
    router_->set_callback(nullptr);
    GuanacoRouterConfig router_config;
    router_config.num_layers = model_config.num_hidden_layers > 0 ? model_config.num_hidden_layers : config_.num_layers;
    router_config.num_experts = model_config.num_experts > 0 ? model_config.num_experts : config_.num_experts;
    router_config.top_k = model_config.num_experts_per_tok > 0 ? model_config.num_experts_per_tok : config_.top_k;
    router_config.on_router_decision = [this](const RouterOutput& output) {
        on_router_output(output);
    };
    
    router_ = std::make_unique<GuanacoRouter>(router_config);
    router_->set_manifest(&manifest_);
    router_->set_cache(cache_.get());
    router_->set_loader(loader_.get());
    
    initialized_ = true;
    std::cout << "[Guanaco] Initialized with " << manifest_.size() 
              << " expert tensors, " << model_config.num_experts << " experts, "
              << "top_k=" << model_config.num_experts_per_tok
              << ", layers=" << model_config.num_hidden_layers << std::endl;
    return true;
}

void GuanacoExecutor::evaluate_token(const std::vector<float>& hidden_states) {
    if (!initialized_) {
        std::cerr << "[Guanaco] Not initialized" << std::endl;
        return;
    }
    
    const MoEModelConfig& model_config = loader_->get_model_config();
    int num_layers = model_config.num_hidden_layers > 0 ? model_config.num_hidden_layers : config_.num_layers;
    
    for (int layer = 0; layer < num_layers; ++layer) {
        evaluate_moe_layer(layer, hidden_states);
    }
}

void GuanacoExecutor::evaluate_moe_layer(int layer_idx, const std::vector<float>& token_hidden_states) {
    const MoEModelConfig& model_config = loader_->get_model_config();
    int num_experts = model_config.num_experts > 0 ? model_config.num_experts : config_.num_experts;
    int top_k = model_config.num_experts_per_tok > 0 ? model_config.num_experts_per_tok : config_.top_k;
    int num_tokens = 1;
    
    std::vector<float> router_logits(num_experts * num_tokens);
    
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < num_experts; ++e) {
            router_logits[t * num_experts + e] = static_cast<float>(e) * 0.1f + t * 0.01f;
        }
    }
    
    router_->intercept_router(layer_idx, router_logits.data(), num_tokens, num_experts);
}

void GuanacoExecutor::set_router_callback(std::function<void(int, const std::vector<int>&)> callback) {
    router_callback_ = std::move(callback);
}

void GuanacoExecutor::on_router_output(const RouterOutput& output) {
    std::cout << "[Guanaco Router] Layer " << output.layer_idx 
              << " selected experts: ";
    for (size_t i = 0; i < output.selected_experts.size(); ++i) {
        std::cout << output.selected_experts[i] << "(w=" << output.expert_weights[i] << ") ";
    }
    std::cout << std::endl;
    
    if (router_callback_) {
        router_callback_(output.layer_idx, output.selected_experts);
    }
}

} // namespace guanaco