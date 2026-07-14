#include "guanaco/guanaco.h"
#include "gguf.h"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace guanaco {

std::vector<ExpertManifestEntry> parse_gguf_expert_manifest(const char* gguf_path) {
    std::vector<ExpertManifestEntry> manifest;
    
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context* ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) {
        return manifest;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t data_offset = gguf_get_data_offset(ctx);
    
    manifest.reserve(n_tensors);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;

        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t byte_size = gguf_get_tensor_size(ctx, i);

        std::string name_str(name);
        int layer_idx = -1;
        int expert_idx = -1;
        int num_experts_in_tensor = 1;
        bool is_expert_tensor = false;
        
        if (name_str.find("ffn_up_exps") != std::string::npos ||
            name_str.find("ffn_down_exps") != std::string::npos ||
            name_str.find("ffn_gate_exps") != std::string::npos ||
            name_str.find(".experts.") != std::string::npos ||
            name_str.find(".exps.") != std::string::npos) {
            
            is_expert_tensor = true;
            
            const char* layer_marker = strstr(name, "blk.");
            if (!layer_marker) layer_marker = strstr(name, "layers.");
            if (layer_marker) {
                char* endptr;
                layer_idx = strtol(layer_marker + 4, &endptr, 10);
            }
            
            const char* expert_marker = strstr(name, ".experts.");
            if (!expert_marker) expert_marker = strstr(name, "exps.");
            if (expert_marker) {
                if (strstr(expert_marker, ".experts.")) {
                    expert_idx = strtol(expert_marker + 9, nullptr, 10);
                } else {
                    expert_idx = strtol(expert_marker + 5, nullptr, 10);
                }
            }
            
            num_experts_in_tensor = 1;
        }

        if (is_expert_tensor && layer_idx >= 0) {
            ExpertManifestEntry entry;
            entry.tensor_name = std::move(name_str);
            entry.layer_idx = layer_idx;
            entry.expert_idx = expert_idx;
            entry.file_offset = data_offset + offset;
            entry.byte_size = byte_size;
            entry.num_experts_in_tensor = num_experts_in_tensor;
            entry.tensor_type = (name_str.find("gate") != std::string::npos) ? 2 : 0;
            entry.quantization_type = ggml_type_name(type);
            manifest.push_back(std::move(entry));
        }
    }

    gguf_free(ctx);
    return manifest;
}

} // namespace guanaco