#include "guanaco/guanaco_affinity.h"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace guanaco {

ExpertAffinityCollector::ExpertAffinityCollector(int num_draft_experts,
                                                 int num_target_experts,
                                                 int num_layers)
    : num_draft_experts_(num_draft_experts)
    , num_target_experts_(num_target_experts)
    , num_layers_(num_layers)
{
    counts_.resize(num_layers_);
    int cells_per_layer = num_draft_experts_ * num_target_experts_;
    for (int l = 0; l < num_layers_; ++l) {
        counts_[l].assign(cells_per_layer, 0);
    }
}

void ExpertAffinityCollector::record(int layer,
                                     const int* draft_ids, int n_draft,
                                     const int* target_ids, int n_target)
{
    if (layer < 0 || layer >= num_layers_) return;
    int64_t* row = counts_[layer].data();
    int stride = num_target_experts_;

    for (int di = 0; di < n_draft; ++di) {
        int d = draft_ids[di];
        if (d < 0 || d >= num_draft_experts_) continue;
        for (int ti = 0; ti < n_target; ++ti) {
            int t = target_ids[ti];
            if (t < 0 || t >= num_target_experts_) continue;
            row[d * stride + t]++;
        }
    }
    ++tokens_;
}

void ExpertAffinityCollector::begin_token()
{
    phase_ = 1;
    draft_buf_layer_.clear();
    draft_buf_ids_.clear();
}

void ExpertAffinityCollector::record_draft(int layer, const int* ids, int n)
{
    if (phase_ != 1) return;
    draft_buf_layer_.push_back(layer);
    draft_buf_ids_.emplace_back(ids, ids + n);
}

void ExpertAffinityCollector::record_target(int layer, const int* ids, int n)
{
    if (phase_ != 2) return;
    // Find matching draft entry by layer
    for (size_t i = 0; i < draft_buf_layer_.size(); ++i) {
        if (draft_buf_layer_[i] == layer) {
            record(layer, draft_buf_ids_[i].data(), (int)draft_buf_ids_[i].size(), ids, n);
            return;
        }
    }
}

void ExpertAffinityCollector::flush_token()
{
    phase_ = 2;
}

std::unique_ptr<ExpertAffinityTable> ExpertAffinityCollector::finalize(float threshold)
{
    auto table = std::make_unique<ExpertAffinityTable>();
    table->threshold = threshold;
    table->layers.resize(num_layers_);
    table->layer_map.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        table->layer_map[l] = l;  // identity by default
    }

    int stride = num_target_experts_;

    for (int l = 0; l < num_layers_; ++l) {
        ExpertAffinityLayer& layer = table->layers[l];
        layer.num_draft_experts  = num_draft_experts_;
        layer.num_target_experts = num_target_experts_;
        layer.ranges.resize(num_draft_experts_ + 1, 0);
        layer.ranges[0] = 0;

        int64_t* row = counts_[l].data();

        for (int d = 0; d < num_draft_experts_; ++d) {
            // Build temporary list of (target_expert, count) above threshold
            int64_t total = 0;
            for (int t = 0; t < num_target_experts_; ++t) {
                total += row[d * stride + t];
            }
            if (total == 0) {
                layer.ranges[d + 1] = (uint32_t)layer.entries.size();
                continue;
            }

            float inv_total = 1.0f / total;
            int start = (int)layer.entries.size();

            for (int t = 0; t < num_target_experts_; ++t) {
                int64_t c = row[d * stride + t];
                if (c == 0) continue;
                float prob = (float)c * inv_total;
                if (prob < threshold) continue;
                layer.entries.push_back({t, prob});
            }

            // Sort by probability descending
            auto begin = layer.entries.begin() + start;
            auto end   = layer.entries.end();
            std::sort(begin, end, [](const auto& a, const auto& b) {
                return a.probability > b.probability;
            });

            layer.ranges[d + 1] = (uint32_t)layer.entries.size();
        }
    }

    return table;
}

int expert_affinity_lookup(const ExpertAffinityTable& table,
                           int draft_layer,
                           const int* draft_ids, int n_draft,
                           int* out, float threshold)
{
    if (draft_layer < 0 || draft_layer >= (int)table.layers.size()) return 0;
    if (threshold < 0) threshold = table.threshold;

    const auto& layer = table.layers[draft_layer];
    int written = 0;

    for (int i = 0; i < n_draft; ++i) {
        int d = draft_ids[i];
        if (d < 0 || d >= layer.num_draft_experts) continue;
        uint32_t begin = layer.ranges[d];
        uint32_t end   = layer.ranges[d + 1];
        for (uint32_t ei = begin; ei < end; ++ei) {
            if (layer.entries[ei].probability < threshold) break;
            out[written++] = layer.entries[ei].target_expert;
        }
    }

    return written;
}

} // namespace guanaco
