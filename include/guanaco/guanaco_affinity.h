#pragma once

#include "guanaco/guanaco.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>

namespace guanaco {

// One predicted target expert, with confidence.
struct GUANACO_API ExpertAffinityEntry {
    int  target_expert;
    float probability;  // 0..1, filtered by threshold
};

// Per-layer correlation: for each draft expert, the set of likely target
// experts sorted descending by probability.
struct GUANACO_API ExpertAffinityLayer {
    // ranges[0 .. num_draft_experts] are prefix-sum indices into entries[].
    // For draft expert d, predictions are:
    //   entries[ranges[d] .. ranges[d+1])
    std::vector<uint32_t>  ranges;
    std::vector<ExpertAffinityEntry> entries;

    int num_draft_experts  = 0;
    int num_target_experts = 0;
};

// The full correlation table, built during warmup and read during inference.
// One instance is shared between the draft and target SteppeLoaders.
struct GUANACO_API ExpertAffinityTable {
    std::vector<ExpertAffinityLayer> layers;  // indexed by draft layer
    // layer_map[target_layer] = draft_layer that predicts it.
    // For MTP (same-model speculation): identity.
    std::vector<int> layer_map;

    float threshold = 0.05f;  // minimum probability to act on
};

// Warmup collector: records draft->target expert co-occurrences during a
// template forward pass, then produces an ExpertAffinityTable.
class GUANACO_API ExpertAffinityCollector {
public:
    ExpertAffinityCollector(int num_draft_experts,
                            int num_target_experts,
                            int num_layers);

    // Record one token's router outputs at the given layer.
    void record(int layer,
                const int* draft_ids, int n_draft,
                const int* target_ids, int n_target);

    // Two-phase recording for warmup
    void begin_token();
    void record_draft(int layer, const int* ids, int n);
    void record_target(int layer, const int* ids, int n);
    void flush_token();

    // Finalize: compute probabilities from raw counts, build the table.
    std::unique_ptr<ExpertAffinityTable> finalize(float threshold = 0.05f);

    int num_tokens() const { return tokens_; }

private:
    int num_draft_experts_;
    int num_target_experts_;
    int num_layers_;
    int tokens_ = 0;

    std::vector<std::vector<int64_t>> counts_;

    int phase_ = 0;
    std::vector<int> draft_buf_layer_;
    std::vector<std::vector<int>> draft_buf_ids_;
};

// Apply the predictions from the table
GUANACO_API int expert_affinity_lookup(const ExpertAffinityTable& table,
                                       int draft_layer,
                                       const int* draft_ids, int n_draft,
                                       int* out, int out_capacity,
                                       float threshold = 0.05f);

} // namespace guanaco
