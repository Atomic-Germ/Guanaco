#include "guanaco/guanaco.h"
#include <algorithm>
#include <iostream>

namespace guanaco {

GuanacoRouter::GuanacoRouter(const GuanacoRouterConfig& config) : config_(config) {}

GuanacoRouter::~GuanacoRouter() = default;

void GuanacoRouter::set_manifest(const std::vector<ExpertManifestEntry>* manifest) {
    manifest_ = manifest;
}

void GuanacoRouter::set_cache(HerdCache* cache) {
    cache_ = cache;
}

void GuanacoRouter::set_loader(SteppeLoader* loader) {
    loader_ = loader;
}

void GuanacoRouter::intercept_router(int layer_idx, const float* router_logits, 
                                      int num_tokens, int num_experts) {
    process_router_output(layer_idx, router_logits, num_tokens, num_experts);
}

void GuanacoRouter::process_router_output(int layer_idx, const float* logits, 
                                           int num_tokens, int num_experts) {
    std::vector<int> selected_experts;
    std::vector<float> expert_weights;
    selected_experts.reserve(config_.top_k * num_tokens);
    expert_weights.reserve(config_.top_k * num_tokens);

    for (int t = 0; t < num_tokens; ++t) {
        std::vector<std::pair<float, int>> scores;
        scores.reserve(num_experts);
        
        const float* token_logits = logits + t * num_experts;
        for (int e = 0; e < num_experts; ++e) {
            scores.emplace_back(token_logits[e], e);
        }
        
        std::partial_sort(scores.begin(), scores.begin() + config_.top_k, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (int k = 0; k < config_.top_k; ++k) {
            selected_experts.push_back(scores[k].second);
            expert_weights.push_back(scores[k].first);
        }
    }

    std::sort(selected_experts.begin(), selected_experts.end());
    selected_experts.erase(std::unique(selected_experts.begin(), selected_experts.end()),
                          selected_experts.end());

    RouterOutput output;
    output.layer_idx = layer_idx;
    output.selected_experts = std::move(selected_experts);
    output.expert_weights = std::move(expert_weights);
    output.num_tokens = num_tokens;
    output.top_k = config_.top_k;

    if (config_.on_router_decision) {
        config_.on_router_decision(output);
    }

    if (cache_ && loader_) {
        prefetch_experts(layer_idx, output.selected_experts);
    }
}

void GuanacoRouter::prefetch_experts(int layer_idx, const std::vector<int>& expert_ids) {
    std::vector<std::future<void>> futures;
    futures.reserve(expert_ids.size());

    for (int expert_id : expert_ids) {
        if (!cache_->is_pinned(layer_idx, expert_id)) {
            auto buf = cache_->allocate_slot(layer_idx, expert_id);
            if (manifest_) {
                for (const auto& entry : *manifest_) {
                    if (entry.layer_idx == layer_idx && entry.expert_idx == expert_id) {
                        auto future = loader_->prefetch_expert(layer_idx, expert_id, buf);
                        futures.push_back(std::move(future));
                        break;
                    }
                }
            }
        }
    }
    
    for (auto& f : futures) {
        f.wait();
    }
}

const std::vector<int>& GuanacoRouter::get_selected_experts() const {
    static std::vector<int> empty;
    return empty;
}

const std::vector<float>& GuanacoRouter::get_expert_weights() const {
    static std::vector<float> empty;
    return empty;
}

std::vector<int> GuanacoRouter::top_k_indices(const float* logits, int num_experts, int top_k) const {
    std::vector<std::pair<float, int>> scores;
    scores.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        scores.emplace_back(logits[e], e);
    }
    std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> result;
    result.reserve(top_k);
    for (int k = 0; k < top_k; ++k) {
        result.push_back(scores[k].second);
    }
    return result;
}

} // namespace guanaco