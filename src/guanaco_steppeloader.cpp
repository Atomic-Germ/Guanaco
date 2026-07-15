#include "guanaco/guanaco.h"
#include "gguf.h"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace guanaco {

SteppeLoader::SteppeLoader(const SteppeLoaderConfig& config) : config_(config) {}

SteppeLoader::~SteppeLoader() {
    shutdown();
}

bool SteppeLoader::initialize() {
    fd_ = open(config_.gguf_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "[Guanaco Storage] Failed to open GGUF file: " << config_.gguf_path << std::endl;
        return false;
    }

    if (!parse_gguf_manifest()) {
        return false;
    }
    
    parse_gguf_metadata();

    return true;
}

void SteppeLoader::shutdown() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool SteppeLoader::parse_gguf_manifest() {
    // First pass: parse tensor dimensions directly from file
    std::unordered_map<std::string, std::array<int64_t, 4>> tensor_dims;
    std::unordered_map<std::string, uint32_t> tensor_n_dims;
    parse_tensor_dims_from_file(tensor_dims, tensor_n_dims);
    
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context* ctx = gguf_init_from_file(config_.gguf_path.c_str(), params);
    if (!ctx) {
        std::cerr << "[Guanaco Storage] Failed to parse GGUF header" << std::endl;
        return false;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t data_offset = gguf_get_data_offset(ctx);
    
    manifest_.reserve(n_tensors);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;

        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t byte_size = gguf_get_tensor_size(ctx, i);

        std::string name_str(name);
        int layer_idx = -1;
        int expert_idx = -1;
        int tensor_type = -1;
        int num_experts_in_tensor = 1;
        
        // Get tensor dimensions if available
        auto dim_it = tensor_dims.find(name_str);
        std::array<int64_t, 4> dims = {1, 1, 1, 1};
        uint32_t n_dims = 1;
        if (dim_it != tensor_dims.end()) {
            dims = dim_it->second;
            n_dims = tensor_n_dims[name_str];
            
            // For expert tensors, the expert count is usually in ne[2] or ne[3]
            for (uint32_t d = 0; d < n_dims; ++d) {
                if (dims[d] > 1 && (d == 2 || d == 3)) {
                    num_experts_in_tensor = static_cast<int>(dims[d]);
                }
            }
        }
        
        if (name_str.find("ffn_up_exps") != std::string::npos ||
            name_str.find("ffn_down_exps") != std::string::npos ||
            name_str.find("ffn_gate_exps") != std::string::npos ||
            name_str.find(".experts.") != std::string::npos ||
            name_str.find(".exps.") != std::string::npos) {
            
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
            
            tensor_type = (name_str.find("gate") != std::string::npos) ? 2 : 0;
        }

        if (layer_idx >= 0 || expert_idx >= 0 || num_experts_in_tensor > 1) {
            ExpertManifestEntry entry;
            entry.tensor_name = std::move(name_str);
            entry.layer_idx = layer_idx;
            entry.expert_idx = expert_idx;
            entry.file_offset = data_offset + offset;
            entry.byte_size = byte_size;
            entry.num_experts_in_tensor = num_experts_in_tensor;
            entry.tensor_type = tensor_type;
            entry.quantization_type = ggml_type_name(type);
            entry.tensor_n_dims = n_dims;
            for (uint32_t d = 0; d < 4; ++d) {
                entry.tensor_dims[d] = dims[d];
            }
            
            // Calculate expert slice info
            if (num_experts_in_tensor > 1) {
                entry.expert_slice_size = byte_size / num_experts_in_tensor;
            }
            
            manifest_.push_back(std::move(entry));
            layer_experts_[layer_idx].push_back(manifest_.back());
        }
    }

    gguf_free(ctx);
    
    std::cout << "[Guanaco Storage] Parsed " << manifest_.size() << " expert tensors across " 
              << layer_experts_.size() << " layers" << std::endl;
    
    return true;
}

void SteppeLoader::parse_tensor_dims_from_file(
    std::unordered_map<std::string, std::array<int64_t, 4>>& out_dims,
    std::unordered_map<std::string, uint32_t>& out_n_dims) {
    // Parse GGUF file directly to get tensor dimensions
    // This mimics the internal gguf.cpp parsing logic
    
    int fd = open(config_.gguf_path.c_str(), O_RDONLY);
    if (fd < 0) return;
    
    // Read header
    char magic[4];
    read(fd, magic, 4);
    uint32_t version;
    read(fd, &version, 4);
    uint64_t n_tensors, n_kv;
    read(fd, &n_tensors, 8);
    read(fd, &n_kv, 8);
    
    // Skip KV pairs
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t key_len;
        read(fd, &key_len, 8);
        lseek(fd, key_len, SEEK_CUR); // skip key
        uint32_t val_type;
        read(fd, &val_type, 4);
        
        // Skip value based on type
        if (val_type == 0 || val_type == 1) { lseek(fd, 1, SEEK_CUR); } // uint8/int8
        else if (val_type == 2 || val_type == 3) { lseek(fd, 2, SEEK_CUR); } // uint16/int16
        else if (val_type == 4 || val_type == 5 || val_type == 6) { lseek(fd, 4, SEEK_CUR); } // uint32/int32/float32
        else if (val_type == 7) { lseek(fd, 1, SEEK_CUR); } // bool
        else if (val_type == 8) { // string
            uint64_t str_len;
            read(fd, &str_len, 8);
            lseek(fd, str_len, SEEK_CUR);
        } else if (val_type == 9) { // array
            uint32_t arr_type;
            uint64_t arr_len;
            read(fd, &arr_type, 4);
            read(fd, &arr_len, 8);
            size_t elem_size = 0;
            if (arr_type == 8) { // string array
                for (uint64_t j = 0; j < arr_len; ++j) {
                    uint64_t s_len;
                    read(fd, &s_len, 8);
                    lseek(fd, s_len, SEEK_CUR);
                }
            } else if (arr_type == 4 || arr_type == 5) { elem_size = 4; }
            else if (arr_type == 10 || arr_type == 11) { elem_size = 8; }
            else if (arr_type == 12) { elem_size = 8; }
            else { elem_size = 4; }
            lseek(fd, arr_len * elem_size, SEEK_CUR);
        } else if (val_type == 10 || val_type == 11) { lseek(fd, 8, SEEK_CUR); } // uint64/int64
        else if (val_type == 12) { lseek(fd, 8, SEEK_CUR); } // float64
    }
    
    // Now at tensor info section
    for (uint64_t i = 0; i < n_tensors; ++i) {
        // Read tensor name
        uint64_t name_len;
        read(fd, &name_len, 8);
        std::string name(name_len, '\0');
        read(fd, name.data(), name_len);
        
        // Read tensor dimensions
        uint32_t n_dims;
        read(fd, &n_dims, 4);
        
        std::array<int64_t, 4> dims = {1, 1, 1, 1};
        for (uint32_t d = 0; d < 4; ++d) {
            if (d < n_dims) {
                int64_t dim;
                read(fd, &dim, 8);
                dims[d] = dim;
            }
        }
        
        // Skip type and offset (we'll read offset from gguf API)
        lseek(fd, 4 + 8, SEEK_CUR); // type (4 bytes) + offset (8 bytes)
        
        out_dims[name] = dims;
        out_n_dims[name] = n_dims;
    }
    
    close(fd);
}

bool SteppeLoader::parse_gguf_metadata() {
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context* ctx = gguf_init_from_file(config_.gguf_path.c_str(), params);
    if (!ctx) {
        std::cerr << "[Guanaco Storage] Failed to parse GGUF header for metadata" << std::endl;
        return false;
    }

    int64_t n_kv = gguf_get_n_kv(ctx);
    
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (!key) continue;
        
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        std::string key_str(key);
        
        // Standard llama.cpp keys
        if (key_str == "llama.expert_count" || key_str == "expert_count" || 
            key_str == "moe.num_experts" || key_str == "num_experts" ||
            key_str.find(".expert_count") != std::string::npos ||
            key_str.find(".expert_used_count") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                if (key_str.find("expert_count") != std::string::npos && key_str.find("used") == std::string::npos) {
                    model_config_.num_experts = gguf_get_val_u32(ctx, i);
                }
            }
        } else if (key_str == "llama.expert_used_count" || key_str == "expert_used_count" ||
                   key_str == "moe.num_experts_per_tok" || key_str == "num_experts_per_tok" ||
                   key_str == "top_k" || key_str.find(".expert_used_count") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.num_experts_per_tok = gguf_get_val_u32(ctx, i);
            }
        } else if (key_str == "llama.block_count" || key_str == "block_count" ||
                   key_str == "num_hidden_layers" || key_str.find(".block_count") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.num_hidden_layers = gguf_get_val_u32(ctx, i);
            }
        } else if (key_str == "llama.embedding_length" || key_str == "embedding_length" ||
                   key_str == "hidden_size" || key_str.find(".embedding_length") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.hidden_size = gguf_get_val_u32(ctx, i);
            }
        } else if (key_str == "general.architecture") {
            if (type == GGUF_TYPE_STRING) {
                model_config_.architecture = gguf_get_val_str(ctx, i);
            }
        }
        
        // Architecture-specific keys
        if (key_str.find("expert_count") != std::string::npos && key_str.find("used") == std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.num_experts = gguf_get_val_u32(ctx, i);
            }
        }
        if (key_str.find("expert_used_count") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.num_experts_per_tok = gguf_get_val_u32(ctx, i);
            }
        }
        if (key_str.find("block_count") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.num_hidden_layers = gguf_get_val_u32(ctx, i);
            }
        }
        if (key_str.find("embedding_length") != std::string::npos) {
            if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
                model_config_.hidden_size = gguf_get_val_u32(ctx, i);
            }
        }
    }

    gguf_free(ctx);
    
    // Fallback: infer from manifest if metadata not found
    if (model_config_.num_experts == 0) {
        std::unordered_set<int> expert_indices;
        for (const auto& entry : manifest_) {
            if (entry.expert_idx >= 0) {
                expert_indices.insert(entry.expert_idx);
            }
        }
        model_config_.num_experts = expert_indices.size() ? *std::max_element(expert_indices.begin(), expert_indices.end()) + 1 : 0;
    }
    
    if (model_config_.num_experts_per_tok == 0 && model_config_.num_experts > 0) {
        // Common defaults
        if (model_config_.num_experts == 8) model_config_.num_experts_per_tok = 2;
        else if (model_config_.num_experts >= 16) model_config_.num_experts_per_tok = 4;
        else if (model_config_.num_experts >= 4) model_config_.num_experts_per_tok = 2;
    }
    
    if (model_config_.num_hidden_layers == 0 && !layer_experts_.empty()) {
        model_config_.num_hidden_layers = layer_experts_.size();
    }

    std::cout << "[Guanaco Storage] Model config: " << model_config_.architecture 
              << ", experts=" << model_config_.num_experts 
              << ", top_k=" << model_config_.num_experts_per_tok
              << ", layers=" << model_config_.num_hidden_layers
              << ", hidden=" << model_config_.hidden_size << std::endl;
    
    return true;
}

std::future<void> SteppeLoader::prefetch_expert(int layer_idx, int expert_idx,
                                                 std::shared_ptr<ExpertTensorBuffer> target_buf) {
    auto it = layer_experts_.find(layer_idx);
    if (it == layer_experts_.end()) {
        auto promise = std::make_shared<std::promise<void>>();
        promise->set_exception(std::make_exception_ptr(std::runtime_error("Layer not found")));
        return promise->get_future();
    }

    for (const auto& entry : it->second) {
        if (entry.expert_idx == expert_idx) {
            return read_expert_async(entry, target_buf);
        }
    }

    auto promise = std::make_shared<std::promise<void>>();
    promise->set_exception(std::make_exception_ptr(std::runtime_error("Expert not found")));
    return promise->get_future();
}

std::future<void> SteppeLoader::prefetch_experts(int layer_idx, const std::vector<int>& expert_indices,
                                                  const std::vector<std::shared_ptr<ExpertTensorBuffer>>& target_bufs) {
    if (expert_indices.size() != target_bufs.size()) {
        auto promise = std::make_shared<std::promise<void>>();
        promise->set_exception(std::make_exception_ptr(std::runtime_error("Size mismatch")));
        return promise->get_future();
    }

    std::vector<std::future<void>> futures;
    futures.reserve(expert_indices.size());

    for (size_t i = 0; i < expert_indices.size(); ++i) {
        futures.push_back(prefetch_expert(layer_idx, expert_indices[i], target_bufs[i]));
    }

    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    std::thread([futures = std::move(futures), promise]() mutable {
        try {
            for (auto& f : futures) {
                f.wait();
            }
            promise->set_value();
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }).detach();

    return future;
}

std::future<void> SteppeLoader::read_expert_async(const ExpertManifestEntry& entry,
                                                   std::shared_ptr<ExpertTensorBuffer> target_buf) {
    // Calculate expert slice size and offset
    size_t slice_size = entry.expert_slice_size > 0 ? entry.expert_slice_size : entry.byte_size;
    size_t slice_offset = entry.file_offset + (entry.expert_idx >= 0 ? static_cast<size_t>(entry.expert_idx) * slice_size : 0);
    
    target_buf->data.resize(slice_size);
    target_buf->expert_id = entry.expert_idx;
    target_buf->layer_idx = entry.layer_idx;
    target_buf->is_ready = false;

    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();

    std::thread([this, slice_offset, slice_size, target_buf, promise]() {
        off_t ret = lseek(fd_, static_cast<off_t>(slice_offset), SEEK_SET);
        if (ret == -1) {
            promise->set_exception(std::make_exception_ptr(std::runtime_error("lseek failed")));
            return;
        }

        size_t total_read = 0;
        while (total_read < slice_size) {
            ssize_t n = read(fd_, target_buf->data.data() + total_read, 
                            slice_size - total_read);
            if (n <= 0) {
                promise->set_exception(std::make_exception_ptr(std::runtime_error("read failed")));
                return;
            }
            total_read += n;
        }

        if (config_.use_madvise && !target_buf->data.empty()) {
            madvise(target_buf->data.data(), target_buf->data.size(), MADV_WILLNEED);
        }

    target_buf->is_ready = true;
    promise->set_value();
    }).detach();

    return future;
}

void SteppeLoader::advise_experts_random() {
    if (fd_ < 0 || manifest_.empty()) {
        return;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        return;
    }

    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    const uintptr_t page_mask = ~((uintptr_t)page_size - 1);

    size_t advised = 0;
    for (const auto& entry : manifest_) {
        // Only expert tensors (fused blocks or individual experts)
        if (entry.expert_idx < 0 && entry.num_experts_in_tensor <= 1) {
            continue;
        }
        if (entry.byte_size == 0) {
            continue;
        }

        uintptr_t region_start = (uintptr_t)addr + entry.file_offset;
        uintptr_t aligned_start = region_start & page_mask;
        size_t size = entry.byte_size + (region_start - aligned_start);

        if (madvise((void*)aligned_start, size, MADV_RANDOM) == 0) {
            ++advised;
        }
    }

    munmap(addr, st.st_size);

    std::cout << "[Guanaco Storage] Applied MADV_RANDOM to " << advised
              << " expert regions" << std::endl;
}

} // namespace guanaco