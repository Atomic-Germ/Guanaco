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
#include <utility>
#include <string>

// imatrix prior: reuse llama.cpp's common loader to read a sibling
// "<model>.imatrix.gguf". It exposes, per expert tensor, a `counts`
// vector sized to the expert dimension.
#include "common/imatrix-loader.h"

namespace guanaco {

SteppeLoader::SteppeLoader(const SteppeLoaderConfig& config) : config_(config) {
#ifdef GUANACO_HAVE_IO_URING
    if (config_.use_io_uring) {
        uring_slab_ = io_uring_slab_create(config_.io_queue_depth);
        if (uring_slab_) {
            std::cerr << "[Guanaco Storage] io_uring prefetch enabled (queue=" << config_.io_queue_depth << ")\n";
        } else {
            std::cerr << "[Guanaco Storage] io_uring init failed; falling back to pread\n";
        }
    }
#endif
}

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
    log_final_summary();

#ifdef GUANACO_HAVE_IO_URING
    if (uring_slab_) {
        io_uring_slab_destroy(uring_slab_);
        uring_slab_ = nullptr;
    }
#endif

    for (auto& kv : expert_tensors_) {
        if (kv.second.slab != nullptr) {
            munmap(kv.second.slab, kv.second.byte_size);
            kv.second.slab = nullptr;
        }
    }
    expert_tensors_.clear();

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
    
    std::cerr << "[Guanaco Storage] Parsed " << manifest_.size() << " expert tensors across " 
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

    std::cerr << "[Guanaco Storage] Model config: " << model_config_.architecture 
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

    std::cerr << "[Guanaco Storage] Applied MADV_RANDOM to " << advised
              << " expert regions" << std::endl;
}

const ExpertManifestEntry* SteppeLoader::lookup_expert(const std::string& name) const {
    for (const auto& e : manifest_) {
        if (e.tensor_name == name) {
            return &e;
        }
    }
    return nullptr;
}

void SteppeLoader::register_expert_tensor(const std::string& name, int layer,
                                            size_t file_offset, size_t byte_size, int num_experts) {
    if (fd_ < 0 || byte_size == 0 || num_experts <= 0) {
        return;
    }
    // Defense in depth: the per-expert slice must divide evenly, otherwise
    // the pread offsets would be misaligned and corrupt adjacent experts.
    // Callers (llama-model.cpp) pre-check this, but skip if it doesn't hold.
    if (byte_size % static_cast<size_t>(num_experts) != 0) {
        return;
    }
    if (expert_tensors_.count(name)) {
        return; // already registered
    }

    ExpertTensor t;
    t.name            = name;
    t.layer           = layer;
    t.file_offset     = file_offset;
    t.byte_size       = byte_size;
    t.num_experts    = num_experts;
    t.per_expert_bytes = byte_size / static_cast<size_t>(num_experts);
    t.loaded.assign(num_experts, false);
    t.total_hits.assign(num_experts, 0);
    t.window_hits.assign(num_experts, 0.0f);
    t.pinned.assign(num_experts, false);
    t.trans.assign((size_t)num_experts * num_experts, 0);

    // Sparse anonymous mapping: only the pages we pread() into ever
    // become resident, so unselected experts cost no RAM.
    void* slab = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (slab == MAP_FAILED) {
        std::cerr << "[Guanaco Storage] Failed to allocate slab for " << name << std::endl;
        return;
    }
    t.slab = slab;

    expert_tensors_.emplace(name, std::move(t));
    std::cerr << "[Guanaco HerdCache] Slab ready for " << name
              << " (" << num_experts << " experts, " << (byte_size / (1024*1024)) << " MB fused)\n";
}

void* SteppeLoader::get_expert_tensor_data(const std::string& name) {
    auto it = expert_tensors_.find(name);
    return it != expert_tensors_.end() ? it->second.slab : nullptr;
}

bool SteppeLoader::pread_full(off_t offset, void* dst, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = pread(fd_, static_cast<char*>(dst) + total, len - total, offset + total);
        if (n <= 0) {
            if (errno == EINTR) continue;
            return false;
        }
        total += static_cast<size_t>(n);
    }
    return true;
}

// After a slice is copied into its slab via pread, advise the kernel to drop
// its copy of the source pages from the file page cache. This is the safe
// analog of colibri's posix_fadvise(DONTNEED): it releases the shared file
// cache, NOT the anonymous slab (MADV_DONTNEED on an anonymous mapping would
// zero our just-loaded weights and corrupt inference).
void SteppeLoader::drop_slice_page_cache(off_t src_offset, size_t len) {
    if (!config_.dontneed || len == 0 || fd_ < 0) return;
    posix_fadvise(fd_, src_offset, (off_t)len, POSIX_FADV_DONTNEED);
}

void SteppeLoader::record_routing(int layer, const int* expert_ids, int n) {
    if (expert_ids == nullptr || n <= 0) return;

    // Learn the per-layer (from->to) transition counts for PILOT. The layer
    // that just fired (pilot_from_layer_) is the "from"; this layer's selected
    // ids are the "to". Layers fire in order within a token, so the previously
    // recorded layer is exactly layer-1.
    if (pilot_from_layer_ == layer - 1 && !pilot_from_ids_.empty()) {
        for (auto& kv : expert_tensors_) {
            ExpertTensor& t = kv.second;
            if (t.layer != layer || t.num_experts <= 0) continue;
            const int E = t.num_experts;
            for (int i = 0; i < n; ++i) {
                int to = expert_ids[i];
                if (to < 0 || to >= E) continue;
                for (int j = 0; j < (int)pilot_from_ids_.size(); ++j) {
                    int from = pilot_from_ids_[j];
                    if (from < 0 || from >= E) continue;
                    ++t.trans[(size_t)from * E + to];
                }
            }
        }
    }

    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.layer != layer || t.num_experts <= 0) continue;

        for (int i = 0; i < n; ++i) {
            int id = expert_ids[i];
            if (id < 0 || id >= t.num_experts) continue;
            ++t.total_hits[id];
            t.window_hits[id] = t.window_hits[id] * kHitDecay_ + 1.0f;
        }
    }

    // Remember this layer's ids so the next layer can learn from->to.
    pilot_from_layer_ = layer;
    pilot_from_ids_.assign(expert_ids, expert_ids + n);

    ++routing_records_;

    if (routing_records_ == kWarmupLayers_) {
        maybe_pin_hot_experts();
    } else if (routing_records_ > kWarmupLayers_ && routing_records_ % kLogEveryLayers_ == 0) {
        maybe_pin_hot_experts();
        log_hot_summary();
    }

    // Periodic pin-cache hit-rate report (independent cadence).
    if (prefetch_calls_ > 0 && prefetch_calls_ % kPinRateEvery_ == 0) {
        log_pin_rate();
    }
}

void SteppeLoader::pilot_prefetch_next_layer(int layer, const int* expert_ids, int n) {
    if (!config_.use_pilot || expert_ids == nullptr || n <= 0) return;
    const int next = layer + 1;
    if (next >= (int)model_config_.num_hidden_layers) return;

    // For each expert selected at this layer, gather the top predicted
    // experts at the next layer from the learned transition matrix, then
    // prefetch them. Hint-only: prefetch_experts() itself skips anything
    // already pinned/loaded, so a wrong guess costs at most a wasted read
    // into a cold slab.
    std::unordered_set<int> predicted;
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.layer != next || t.num_experts <= 0) continue;
        const int E = t.num_experts;

        // Accumulate predicted-next scores: sum of transition counts over all
        // selected "from" experts at this layer.
        std::vector<uint64_t> score(E, 0);
        bool any = false;
        for (int i = 0; i < n; ++i) {
            int from = expert_ids[i];
            if (from < 0 || from >= E) continue;
            const uint64_t* row = &t.trans[(size_t)from * E];
            for (int to = 0; to < E; ++to) {
                if (row[to]) { score[to] += row[to]; any = true; }
            }
        }
        if (!any) continue;  // not enough history yet; skip

        // Keep the top-K most-likely next experts (bounded by pin budget).
        const int K = config_.max_active_experts;
        std::vector<int> order(E);
        for (int i = 0; i < E; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return score[a] > score[b];
        });
        for (int r = 0; r < K && r < E; ++r) {
            predicted.insert(order[r]);
        }
    }
    if (predicted.empty()) return;

    ++pilot_calls_;
    pilot_slice_reads_ += predicted.size();
    std::vector<int> ids(predicted.begin(), predicted.end());
    // Reuse the same prefetch path; it classifies hits/misses and copies.
    prefetch_experts(next, ids.data(), (int)ids.size());
}

void SteppeLoader::maybe_pin_hot_experts() {
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.num_experts <= 0) continue;

        // Rank expert windowed hotness for this tensor.
        std::vector<int> order(t.num_experts);
        for (int i = 0; i < t.num_experts; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return t.window_hits[a] > t.window_hits[b];
        });

        // Pin the top-K experts (bounded by config), skipping those already
        // pinned. A pinned expert is read once (below) and never evicted.
        const int topk = config_.max_active_experts;
        for (int rank = 0; rank < topk && rank < t.num_experts; ++rank) {
            int id = order[rank];
            if (t.pinned[id]) continue;
            if (t.window_hits[id] <= 0.0f) break;  // no usage yet

            if (t.slab && !t.loaded[id]) {
                const off_t src = static_cast<off_t>(t.file_offset) + static_cast<off_t>(id) * t.per_expert_bytes;
                void* dst = static_cast<char*>(t.slab) + static_cast<size_t>(id) * t.per_expert_bytes;
#ifdef GUANACO_HAVE_IO_URING
                if (uring_slab_) {
                    std::vector<IoUringSlice> one = { { dst, t.per_expert_bytes, static_cast<int64_t>(src) } };
                    if (!io_uring_slab_read(uring_slab_, fd_, one)) {
                        if (!pread_full(src, dst, t.per_expert_bytes)) continue;
                    }
                } else
#endif
                {
                    if (!pread_full(src, dst, t.per_expert_bytes)) continue;
                }
                t.loaded[id] = true;
                drop_slice_page_cache(src, t.per_expert_bytes);
            }
            t.pinned[id] = true;
            ++pinned_total_;
            ++pinned_current_;
        }
    }

    if (routing_records_ >= kWarmupLayers_ && routing_records_ % kLogEveryLayers_ == 0) {
        std::cerr << "[Guanaco HerdCache] Hot experts pinned: " << pinned_total_
                  << " total across " << expert_tensors_.size() << " tensors\n";
    }
}

size_t SteppeLoader::load_imatrix_prior(const std::string& model_path) {
    if (imatrix_seeded_ || expert_tensors_.empty()) {
        return 0;
    }
    imatrix_seeded_ = true;  // don't retry

    if (!config_.use_imatrix) {
        std::cerr << "[Guanaco HerdCache] imatrix prior disabled (GUANACO_IMATRIX=0); "
                  << "cold-start pinning will use runtime stats.\n";
        return 0;
    }

    // Resolve the sibling imatrix file: "<path>.imatrix.gguf".
    // Mirror the convention used by llama.cpp's imatrix/quantize tools
    // (e.g. model.gguf -> model.gguf.imatrix.gguf).
    std::string imatrix_path = model_path + ".imatrix.gguf";

    struct stat st{};
    if (stat(imatrix_path.c_str(), &st) != 0) {
        std::cerr << "[Guanaco HerdCache] No imatrix prior found ("
                  << imatrix_path << "); cold-start pinning will use runtime stats.\n";
        imatrix_seeded_ = true;  // don't retry
        return 0;
    }

    common_imatrix imatrix{};
    if (!common_imatrix_load(imatrix_path, imatrix)) {
        std::cerr << "[Guanaco HerdCache] Failed to parse imatrix prior: "
                  << imatrix_path << "\n";
        imatrix_seeded_ = true;
        return 0;
    }

    size_t seeded = 0;
    uint64_t total_observations = 0;
    for (const auto& kv : imatrix.entries) {
        const std::string& entry_name = kv.first;
        const common_imatrix_entry& entry = kv.second;
        auto it = expert_tensors_.find(entry_name);
        if (it == expert_tensors_.end()) {
            continue;  // not a fused expert tensor we redirect
        }
        ExpertTensor& t = it->second;
        const int n = std::min<int>((int)entry.counts.size(), t.num_experts);
        for (int ex = 0; ex < n; ++ex) {
            const int64_t c = entry.counts[ex];
            if (c <= 0) continue;
            // Seed the durable total; this is the calibration-time selection
            // frequency and directly drives the initial pin pass below.
            t.total_hits[ex] += static_cast<uint64_t>(c);
            total_observations += static_cast<uint64_t>(c);
            ++seeded;
        }
    }

    std::cerr << "[Guanaco HerdCache] imatrix prior loaded: " << imatrix_path
              << " (" << total_observations << " expert selections across "
              << seeded << " expert slots)\n";

    // Immediately pin the calibration-hot experts so they are resident from
    // token 0, before any runtime routing statistics exist.
    if (seeded > 0) {
        pin_from_seeded_totals();
    }
    return seeded;
}

void SteppeLoader::pin_from_seeded_totals() {
    size_t pinned_before = pinned_total_;
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.num_experts <= 0) continue;

        // Rank by seeded total (calibration frequency); fall back to the
        // EWMA window if runtime stats exist.
        std::vector<int> order(t.num_experts);
        for (int i = 0; i < t.num_experts; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return t.total_hits[a] > t.total_hits[b];
        });

        const int topk = config_.max_active_experts;
        for (int rank = 0; rank < topk && rank < t.num_experts; ++rank) {
            int id = order[rank];
            if (t.pinned[id]) continue;
            if (t.total_hits[id] <= 0) break;  // no calibration usage

            if (t.slab && !t.loaded[id]) {
                const off_t src = static_cast<off_t>(t.file_offset) + static_cast<off_t>(id) * t.per_expert_bytes;
                void* dst = static_cast<char*>(t.slab) + static_cast<size_t>(id) * t.per_expert_bytes;
#ifdef GUANACO_HAVE_IO_URING
                if (uring_slab_) {
                    std::vector<IoUringSlice> one = { { dst, t.per_expert_bytes, static_cast<int64_t>(src) } };
                    if (!io_uring_slab_read(uring_slab_, fd_, one)) {
                        if (!pread_full(src, dst, t.per_expert_bytes)) continue;
                    }
                } else
#endif
                {
                    if (!pread_full(src, dst, t.per_expert_bytes)) continue;
                }
                t.loaded[id] = true;
                drop_slice_page_cache(src, t.per_expert_bytes);
            }
            t.pinned[id] = true;
            ++pinned_total_;
            ++pinned_current_;
        }
    }

    if (pinned_total_ > pinned_before) {
        std::cerr << "[Guanaco HerdCache] imatrix prior pinned "
                  << (pinned_total_ - pinned_before)
                  << " experts (" << pinned_total_ << " total across "
                  << expert_tensors_.size() << " tensors)\n";
    }
}

void SteppeLoader::log_hot_summary() {    // Report the single hottest expert per layer (cheap, informative).
    std::cerr << "[Guanaco HerdCache] Routing hotness (layer:expert window_hits):";
    int printed = 0;
    for (auto& kv : expert_tensors_) {
        const ExpertTensor& t = kv.second;
        if (t.num_experts <= 0) continue;
        int best = 0;
        for (int i = 1; i < t.num_experts; ++i) {
            if (t.window_hits[i] > t.window_hits[best]) best = i;
        }
        if (t.window_hits[best] <= 0.0f) continue;
        std::cerr << " L" << t.layer << ":" << best << "=" << (int)t.window_hits[best];
        if (++printed >= 8) break;  // keep the line short
    }
    std::cerr << std::endl;
}

void SteppeLoader::log_pin_rate() {
    const uint64_t total = pin_hits_ + warm_hits_ + disk_miss_;
    if (total == 0) return;
    const uint64_t resident = pin_hits_ + warm_hits_;  // no disk read needed
    const int pct = (int)(100ULL * resident / total);
    const int ppct = (int)(100ULL * pin_hits_ / total);  // fully pinned (hot) share
    std::cerr << "[Guanaco HerdCache] pin cache: " << pct << "% resident ("
              << ppct << "% hot-pinned), " << resident << "/" << total
              << " expert accesses, pinned=" << pinned_total_
              << " across " << expert_tensors_.size() << " tensors\n";
}

void SteppeLoader::log_final_summary() {
    // Only emit after real work; the hook is also destroyed transiently
    // during model-fit before any expert runs, which would print noise.
    if (prefetch_calls_ == 0) return;

    const uint64_t total = pin_hits_ + warm_hits_ + disk_miss_;
    const uint64_t resident = pin_hits_ + warm_hits_;
    const int pct = total ? (int)(100ULL * resident / total) : 0;
    const int ppct = total ? (int)(100ULL * pin_hits_ / total) : 0;

    const size_t n_tensors = expert_tensors_.size();
    const int per_tensor = n_tensors ? (int)(pinned_current_ / n_tensors) : 0;

    std::cerr << "[Guanaco HerdCache] final: pin budget="
              << config_.max_active_experts
              << " per tensor, experts pinned=" << pinned_current_
              << " (" << per_tensor << " avg / tensor across " << n_tensors
              << " tensors)"
              << ", pin cache=" << pct << "% resident (" << ppct
              << "% hot-pinned)"
              << ", " << total << " expert accesses over "
              << prefetch_calls_ << " prefetch calls"
              << (config_.use_pilot
                  ? (", pilot: " + std::to_string(pilot_calls_) + " calls / " + std::to_string(pilot_slice_reads_) + " slices")
                  : "")
              << (imatrix_seeded_ ? ", imatrix prior used" : "")
              << "\n";
}

std::vector<SteppeLoader::HotExpertStats> SteppeLoader::get_hot_expert_stats() const {
    std::vector<HotExpertStats> out;
    for (const auto& kv : expert_tensors_) {
        const ExpertTensor& t = kv.second;
        for (int id = 0; id < t.num_experts; ++id) {
            if (t.pinned[id] || t.window_hits[id] > 0.0f) {
                HotExpertStats s;
                s.layer       = t.layer;
                s.expert      = id;
                s.total_hits  = t.total_hits[id];
                s.window_hits = t.window_hits[id];
                s.pinned      = t.pinned[id];
                out.push_back(s);
            }
        }
    }
    return out;
}

void SteppeLoader::prefetch_experts(int layer, const int* expert_ids, int n) {
    if (fd_ < 0 || expert_ids == nullptr || n <= 0) {
        return;
    }

    ++prefetch_calls_;

#ifdef GUANACO_HAVE_IO_URING
    std::vector<IoUringSlice> slices;
    std::vector<std::pair<ExpertTensor*, int>> slice_owners;
#endif

    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.layer != layer || t.slab == nullptr || t.per_expert_bytes == 0) {
            continue;
        }

        // Classify each router-selected expert for the pin-cache metric,
        // then deduplicate the ones that still need a disk read.
        //  - pinned: hot expert, already resident, zero disk I/O
        //  - loaded: warm slice from a previous read, still resident
        //  - otherwise: requires a fresh disk read this call
        std::unordered_set<int> want;
        for (int i = 0; i < n; ++i) {
            int id = expert_ids[i];
            if (id < 0 || id >= t.num_experts) continue;
            if (t.pinned[id]) {
                ++pin_hits_;
            } else if (t.loaded[id]) {
                ++warm_hits_;
            } else {
                ++disk_miss_;
                want.insert(id);
            }
        }
        if (want.empty()) {
            continue;
        }

        for (int id : want) {
            const off_t src = static_cast<off_t>(t.file_offset) + static_cast<off_t>(id) * t.per_expert_bytes;
            void*  dst = static_cast<char*>(t.slab) + static_cast<size_t>(id) * t.per_expert_bytes;
#ifdef GUANACO_HAVE_IO_URING
            if (uring_slab_) {
                slices.push_back({ dst, t.per_expert_bytes, static_cast<int64_t>(src) });
                slice_owners.push_back({ &t, id });
            } else
#endif
            {
                if (pread_full(src, dst, t.per_expert_bytes)) {
                    t.loaded[id] = true;
                    drop_slice_page_cache(src, t.per_expert_bytes);
                }
            }
        }
    }

#ifdef GUANACO_HAVE_IO_URING
    if (uring_slab_ && !slices.empty()) {
        if (io_uring_slab_read(uring_slab_, fd_, slices)) {
            for (auto& p : slice_owners) {
                ExpertTensor& t = *p.first;
                int id = p.second;
                const off_t src = static_cast<off_t>(t.file_offset) + static_cast<off_t>(id) * t.per_expert_bytes;
                p.first->loaded[p.second] = true;
                drop_slice_page_cache(src, t.per_expert_bytes);
            }
        } else {
            // Fallback: read each slice synchronously.
            for (auto& p : slice_owners) {
                ExpertTensor& t = *p.first;
                int id = p.second;
                const off_t src = static_cast<off_t>(t.file_offset) + static_cast<off_t>(id) * t.per_expert_bytes;
                void* dst = static_cast<char*>(t.slab) + static_cast<size_t>(id) * t.per_expert_bytes;
                if (pread_full(src, dst, t.per_expert_bytes)) {
                    t.loaded[id] = true;
                    drop_slice_page_cache(src, t.per_expert_bytes);
                }
            }
        }
    }
#endif
}

} // namespace guanaco