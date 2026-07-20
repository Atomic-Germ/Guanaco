#include "guanaco/guanaco.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <glob.h>
#include <dirent.h>
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

    // Parse metadata first so we know the authoritative MoE expert count and
    // can disable streaming for dense models before scanning the manifest.
    parse_gguf_metadata();

    if (!parse_gguf_manifest()) {
        return false;
    }

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
        // Residency of the mmap pages is owned by the OS; nothing to free here.
        kv.second.mmap_base = nullptr;
    }
    expert_tensors_.clear();

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

// Build the list of GGUF shard paths to scan. A sharded model's
// filename looks like "...-00001-of-00004.gguf"; we glob the sibling
// shards (00001..00004) so expert tensors living in any shard are
// discovered. Single-file models just return the one path.
static std::vector<std::string> build_shard_paths(const std::string& path) {
    std::vector<std::string> out;
    out.push_back(path);

    // Split into directory + basename; all shard-name matching is done on the
    // basename (the directory is just prepended back when building full paths).
    std::string dir = path.substr(0, path.find_last_of('/') + 1);
    std::string base = path.substr(path.find_last_of('/') + 1);

    // Match "...-<cur>-of-<total>.gguf" within the basename
    const char* dash = nullptr;
    const char* of = nullptr;
    const char* dot = strstr(base.c_str(), ".gguf");
    if (dot) {
        // scan backwards for "-of-"
        for (const char* p = dot; p > base.c_str(); --p) {
            if (strncmp(p, "-of-", 4) == 0) { of = p; break; }
        }
    }
    if (of) {
        // of points at the "-of-" separator. Walk back over the current shard
        // NUMBER digits, then the dash immediately before them is the one that
        // separates the model basename from the shard index (e.g. "...Q8_0-00001-of-...").
        const char* p = of - 1;
        while (p > base.c_str() && *p >= '0' && *p <= '9') --p;
        if (*p == '-') dash = p;
    }
    if (!dash || !of) return out;  // not a sharded name

    // prefix is the basename up to and including the dash before the shard
    // number (e.g. "NVIDIA-...-Q8_0-"); suffix is "-of-NNNN.gguf" onward.
    const std::string prefix(base.c_str(), dash - base.c_str() + 1);
    const std::string suffix(of);  // "-of-NNNN.gguf" onward (starts at the '-')
    int total = atoi(of + 4);
    if (total <= 1) return out;

    DIR* d = opendir(dir.c_str());
    if (d != nullptr) {
        struct dirent* de = nullptr;
        while ((de = readdir(d)) != nullptr) {
            std::string nm(de->d_name);
            // Skip "."/".." and any name too short to possibly match the
            // suffix (avoids unsigned underflow in the compare below).
            if (nm.size() < suffix.size()) continue;
            // Compare against the full prefix (which includes the trailing
            // dash); sibling names are "<prefix><shard>-of-<total>.gguf".
            bool pm = (nm.compare(0, prefix.size(), prefix) == 0);
            bool sm = (nm.compare(nm.size() - suffix.size(), suffix.size(), suffix) == 0);
            std::string full = dir + nm;
            if (pm && sm && full != path) out.push_back(std::move(full));
        }
        closedir(d);
    }
    (void)total;
    return out;
}

// Read only a GGUF shard's metadata (header + KV + tensor-info table) into a
// small heap buffer and build a gguf_context from it. This deliberately avoids
// gguf_init_from_file, which mmaps the entire (potentially huge) tensor data
// blob - we only need tensor names/offsets/dims, so we grow a read buffer until
// it covers the whole metadata region (gguf_get_data_offset <= buffer size).
// Returns a context the caller must gguf_free, or nullptr on failure.
static struct gguf_context* read_gguf_metadata_ctx(const std::string& shard) {
    int fd = open(shard.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;

    std::vector<uint8_t> buf;
    struct gguf_context* ctx = nullptr;
    size_t cap = 1 << 20;  // start at 1 MiB, grow as needed
    const size_t max_cap = 64 << 20;  // metadata table should never approach this
    bool ok = false;
    for (;;) {
        buf.resize(cap);
        ssize_t rd = pread(fd, buf.data(), cap, 0);
        if (rd < 0 || (size_t)rd < cap) break;
        struct gguf_init_params params = { false, nullptr };
        ctx = gguf_init_from_buffer(buf.data(), cap, params);
        if (!ctx) {
            // A null return on a valid file means the buffer was too small to
            // hold the full metadata (gguf aborts the parse rather than
            // returning a partial ctx). Grow and retry; only give up if we've
            // hit the cap. This keeps the read metadata-only (no full mmap).
            if (cap >= max_cap) break;
            cap *= 2;
            continue;
        }
        // data_offset is the end of the metadata region; if it lies beyond what
        // we read, the metadata is larger than our buffer - grow and retry.
        if (gguf_get_data_offset(ctx) <= cap) { ok = true; break; }
        gguf_free(ctx);
        ctx = nullptr;
        if (cap >= max_cap) break;
        cap *= 2;
    }
    close(fd);
    if (!ok && ctx) { gguf_free(ctx); ctx = nullptr; }
    return ctx;
}

bool SteppeLoader::parse_gguf_manifest() {
    // Dense model: nothing to stream. Skip the whole scan (and avoid emitting
    // the misleading "Parsed N expert tensors" line for a non-MoE model).
    if (!enabled_) {
        std::cerr << "[Guanaco Storage] Skipping expert manifest (streaming disabled)\n";
        return true;
    }

    // Scan every shard, not just shard 1. A sharded GGUF splits the
    // expert tensors across shards, so reading only the first shard would
    // miss most of them (e.g. a 4-shard 120B model registering 0
    // expert tensors and silently falling back to a plain mmap).
    // Each shard's metadata is read with a small pread (KB-MB), never a full
    // mmap, so discovering all shards costs almost no RAM.
    std::vector<std::string> shards = build_shard_paths(config_.gguf_path);

    const int model_experts = model_config_.num_experts;
    for (const auto& shard : shards) {
        struct gguf_context* ctx = read_gguf_metadata_ctx(shard);
        if (!ctx) continue;

        int64_t n_tensors = gguf_get_n_tensors(ctx);
        size_t data_offset = gguf_get_data_offset(ctx);

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(ctx, i);
            if (!name) continue;

            enum ggml_type type = gguf_get_tensor_type(ctx, i);
            size_t offset = gguf_get_tensor_offset(ctx, i);
            size_t byte_size = gguf_get_tensor_size(ctx, i);
            const int64_t* ne = gguf_get_tensor_ne(ctx, i);

            std::string name_str(name);
            int layer_idx = -1;
            int expert_idx = -1;
            int tensor_type = -1;
            int num_experts_in_tensor = 1;

            // Expert count is the trailing (highest-indexed) dim > 1.
            if (ne) {
                for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
                    if (ne[d] > 1) {
                        num_experts_in_tensor = static_cast<int>(ne[d]);
                        break;
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

            // Only treat a tensor as a real MoE expert block if its trailing
            // dim is a genuine expert count. A dense FFN tensor (trailing
            // dim = hidden_size) would otherwise be mis-parsed as one giant
            // "expert" (thrashing / OOM on DeepSeek-70B, Seed-OSS).
            if (num_experts_in_tensor > 1 && model_experts > 1 &&
                num_experts_in_tensor != model_experts) {
                continue;
            }

            if (layer_idx >= 0 || expert_idx >= 0 || num_experts_in_tensor > 1) {
                ExpertManifestEntry entry;
                entry.tensor_name = std::move(name_str);
                entry.layer_idx = layer_idx;
                entry.expert_idx = expert_idx;
                // file_offset is relative to THIS shard's data start.
                entry.file_offset = data_offset + offset;
                entry.byte_size = byte_size;
                entry.num_experts_in_tensor = num_experts_in_tensor;
                entry.tensor_type = tensor_type;
                entry.quantization_type = ggml_type_name(type);
                if (ne) {
                    for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
                        entry.tensor_dims[d] = ne[d];
                    }
                }
                entry.tensor_n_dims = 0;
                if (ne) {
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        if (ne[d] > 1) entry.tensor_n_dims = d + 1;
                    }
                }

                if (num_experts_in_tensor > 1) {
                    entry.expert_slice_size = byte_size / num_experts_in_tensor;
                }

                manifest_.push_back(std::move(entry));
                layer_experts_[layer_idx].push_back(manifest_.back());
            }
        }

        gguf_free(ctx);
    }

    std::cerr << "[Guanaco Storage] Parsed " << manifest_.size() << " expert tensors across "
              << layer_experts_.size() << " layers (from " << shards.size() << " shard(s))\n";

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

    // Dense model (or expert count unknown): there is nothing to stream.
    // Disable streaming so we never mistake a dense FFN tensor for a 1-expert
    // block and thrash / corrupt the run. ggml's normal mmap is used instead.
    if (model_config_.num_experts <= 1) {
        enabled_ = false;
        std::cerr << "[Guanaco Storage] No MoE experts detected (experts="
                  << model_config_.num_experts << ") - disabling Guanaco disk streaming (passthrough)\n";
    }

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
    if (fd_ < 0 || expert_tensors_.empty()) {
        return;
    }

    // Tell the kernel not to do sequential read-ahead on the expert mmap
    // regions (MADV_RANDOM), and not to cache them sequentially from the fd
    // (POSIX_FADV_RANDOM). This keeps the OS from aggressively faulting in
    // adjacent experts we will not use, which is the whole point of routing.
    size_t advised = 0;
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.mmap_base == nullptr || t.byte_size == 0) continue;
        madvise(t.mmap_base, t.byte_size, MADV_RANDOM);
        if (posix_fadvise(fd_, (off_t)t.file_offset, (off_t)t.byte_size, POSIX_FADV_RANDOM) == 0) {
            ++advised;
        }
    }

    std::cerr << "[Guanaco Storage] Applied MADV_RANDOM / POSIX_FADV_RANDOM to " << advised
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
                                            size_t file_offset, size_t byte_size, int num_experts,
                                            void* mmap_base) {
    if (!enabled_ || fd_ < 0 || byte_size == 0 || num_experts <= 1) {
        return;
    }
    // Defense in depth: the per-expert slice must divide evenly, otherwise
    // the slice offsets would be misaligned and corrupt adjacent experts.
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
    t.mmap_base      = mmap_base;
    t.loaded.assign(num_experts, true);   // mapped resident at load time
    t.total_hits.assign(num_experts, 0);
    t.window_hits.assign(num_experts, 0.0f);
    t.pinned.assign(num_experts, false);
    t.last_used.assign(num_experts, 0);
    t.trans.assign((size_t)num_experts * num_experts, 0);
    t.budget = (int)config_.max_active_experts;  // initial default per-tensor cap

    expert_tensors_.emplace(name, std::move(t));

    // Track the set of MoE layer indices (sorted, unique) so pilot can predict
    // for the next MoE layer even when MoE blocks are sparse / non-contiguous.
    if (std::find(moe_layers_.begin(), moe_layers_.end(), layer) == moe_layers_.end()) {
        auto it = std::lower_bound(moe_layers_.begin(), moe_layers_.end(), layer);
        moe_layers_.insert(it, layer);
    }

    std::cerr << "[Guanaco HerdCache] Tracking " << name
              << " (" << num_experts << " experts, " << (byte_size / (1024*1024)) << " MB fused)\n";
}

 void* SteppeLoader::get_expert_tensor_data(const std::string& name) {
     auto it = expert_tensors_.find(name);
     return it != expert_tensors_.end() ? it->second.mmap_base : nullptr;
 }

 void SteppeLoader::record_routing(int layer, const int* expert_ids, int n) {
    if (!enabled_ || expert_ids == nullptr || n <= 0) return;

    // Learn the per-layer (from->to) transition counts for PILOT. The
    // previously-fired MoE layer (pilot_from_layer_) is the "from"; this
    // layer's selected ids are the "to". Layers fire in strictly increasing
    // order within a token, so a strictly-increasing layer means we are still
    // in the same token and can learn the transition. A non-increasing layer
    // means a new token started, so we reset without learning the bogus
    // wrap-around transition. This also works for sparse-MoE models where
    // not every layer has routed experts: the gap between consecutive fired
    // layers is absorbed rather than breaking the adjacency check.
    if (pilot_from_layer_ >= 0 && layer > pilot_from_layer_ && !pilot_from_ids_.empty()) {
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
     if (!enabled_ || !config_.use_pilot || expert_ids == nullptr || n <= 0) return;
     // Sparse MoE: not every layer is a routed MoE block, and the MoE layers
     // are not contiguous (e.g. nemotron_h_moe / afmoe sit at blk.1,3,6,8...).
     // Predict for the NEXT MoE layer after this one, not layer+1, which would
     // skip over the dense layers and never match a tracked expert tensor.
     int next = -1;
     for (int L : moe_layers_) {
         if (L > layer) { next = L; break; }
     }
     if (next < 0) return;

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

        // Keep the top-K most-likely next experts (bounded by this tensor's
        // dynamic budget).
        const int K = (t.budget > 0 ? t.budget : (int)config_.max_active_experts);
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

void SteppeLoader::touch(ExpertTensor& t, int id) {
    if (id >= 0 && id < t.num_experts) {
        t.last_used[id] = ++eclock_;
    }
}

void SteppeLoader::evict_to_budget(ExpertTensor& t) {
    const int budget = t.budget > 0 ? t.budget : (int)config_.max_active_experts;
    if (budget <= 0 || t.mmap_base == nullptr) return;

    // Resident = pinned OR loaded (warm). Pinned are exempt from eviction.
    int resident = 0;
    for (int i = 0; i < t.num_experts; ++i) {
        if (t.pinned[i] || t.loaded[i]) ++resident;
    }
    if (resident <= budget) return;

    // Evict least-recently-used non-pinned loaded slices until within budget.
    // Eviction drops the mmap pages (MADV_DONTNEED): the OS frees the RAM and
    // a later access by mul_mat_id transparently page-faults the slice back
    // from the GGUF file. The bytes are identical (same file-backed address),
    // so there is no correctness risk - only a disk read on next use.
    //
    // Two-pass eviction: PASS A only reclaims COLD slices (windowed hotness
    // below kHotProtect_), protecting hot experts from pure-LRU thrash even when
    // their last_used tick has gone stale. PASS B is a fallback that, if we are
    // still over budget after cold slices are gone, evicts LRU among ALL
    // non-pinned slices (hot included) so we always meet the budget. This can
    // only reduce evictions of useful experts versus the old single-pass LRU.
    int to_evict = resident - budget;

    auto reclaim = [&](bool cold_only) -> int {
        int reclaimed = 0;
        while (to_evict > 0) {
            int victim = -1;
            uint64_t oldest = UINT64_MAX;
            for (int i = 0; i < t.num_experts; ++i) {
                if (t.pinned[i] || !t.loaded[i]) continue;
                if (cold_only && t.window_hits[i] > kHotProtect_) continue;
                if (t.last_used[i] < oldest) { oldest = t.last_used[i]; victim = i; }
            }
            if (victim < 0) break;  // nothing left to evict in this pass

            ++evictions_;
            // Tuning instrumentation: tally how many were "hot" (a hot-protect
            // policy keeps these in PASS A; only PASS B may still drop them).
            if (t.window_hits[victim] > kHotProtect_) ++evict_hot_;
            if (evict_log_.size() >= kEvictLogMax_) evict_log_.erase(evict_log_.begin());
            evict_log_.push_back({&t, victim, eclock_, false});
            // Gate: loaded[victim]=false means the next select re-faults it.
            t.loaded[victim] = false;
            void* addr = static_cast<char*>(t.mmap_base) + (size_t)victim * t.per_expert_bytes;
            madvise(addr, t.per_expert_bytes, MADV_DONTNEED);
            --to_evict;
            ++reclaimed;
        }
        return reclaimed;
    };

    reclaim(true);    // PASS A: cold slices only, protect hot experts
    reclaim(false);   // PASS B: fallback, evict LRU among all non-pinned
}

void SteppeLoader::recompute_budgets() {
    const int global = (int)config_.max_active_experts;
    if (global <= 0 || expert_tensors_.empty()) return;

    // Count distinct (ever-used) experts per tensor as the demand signal.
    std::vector<std::pair<ExpertTensor*, int>> demand;
    demand.reserve(expert_tensors_.size());
    int max_distinct = 1;
    long total_demand = 0;
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        int distinct = 0;
        for (int i = 0; i < t.num_experts; ++i) {
            if (t.total_hits[i] > 0) ++distinct;
        }
        // A tensor with zero runtime history still earns a floor of 1 so it
        // can hold its top expert if routing later activates it.
        if (distinct == 0) distinct = 1;
        demand.push_back({&t, distinct});
        if (distinct > max_distinct) max_distinct = distinct;
        total_demand += distinct;
    }

    // Allocate each tensor a share of the global per-tensor cap proportional
    // to its distinct-expert demand, clamped to [1, global]. This keeps the
    // aggregate at or below global * n_tensors while steering slots to the
    // tensors that actually cycle through many experts.
    const int n = (int)demand.size();
    const long total_cap = (long)global * n;
    for (auto& d : demand) {
        int share = (int)((long)d.second * total_cap / total_demand);
        if (share < 1) share = 1;
        if (share > global) share = global;
        d.first->budget = share;
    }
}

void SteppeLoader::maybe_pin_hot_experts() {
    recompute_budgets();
    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.num_experts <= 0 || t.mmap_base == nullptr) continue;

        // Rank expert windowed hotness for this tensor.
        std::vector<int> order(t.num_experts);
        for (int i = 0; i < t.num_experts; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return t.window_hits[a] > t.window_hits[b];
        });

        // Pin the top-K experts (bounded by this tensor's dynamic budget),
        // skipping those already pinned. A pinned expert is faulted in
        // (MADV_WILLNEED) and never evicted, so its slice stays resident.
        const int topk = (t.budget > 0 ? t.budget : (int)config_.max_active_experts);
        for (int rank = 0; rank < topk && rank < t.num_experts; ++rank) {
            int id = order[rank];
            if (t.pinned[id]) continue;
            if (t.window_hits[id] <= 0.0f) break;  // no usage yet

            if (!t.loaded[id]) {
                // Ensure the slice is resident (it may have been evicted).
                void* addr = static_cast<char*>(t.mmap_base) + (size_t)id * t.per_expert_bytes;
                madvise(addr, t.per_expert_bytes, MADV_WILLNEED);
                t.loaded[id] = true;
            }
            t.pinned[id] = true;
            ++pinned_total_;
            ++pinned_current_;
        }

        // After refreshing this tensor's hot pins, drop its cold (non-pinned)
        // slices down to the resident budget. Pinned experts are exempt, and
        // the slices we just WILLNEED'd are recent so they survive the LRU cut.
        evict_to_budget(t);
    }

    if (routing_records_ >= kWarmupLayers_ && routing_records_ % kLogEveryLayers_ == 0) {
        std::cerr << "[Guanaco HerdCache] distinct experts pinned: " << pinned_total_
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

    // Resolve the sibling imatrix file. First try the exact convention used by
    // llama.cpp's imatrix/quantize tools (model.gguf -> model.gguf.imatrix.gguf),
    // then fall back to scanning the GGUF's directory for any *imatrix*.gguf
    // that shares the model's base name (e.g. model.imatrix.gguf, or a
    // modelname-Q8_0.imatrix.gguf sitting next to the shard).
    std::vector<std::string> candidates;
    candidates.push_back(model_path + ".imatrix.gguf");

    {
        std::string dir = model_path.substr(0, model_path.find_last_of('/') + 1);
        std::string base = model_path.substr(model_path.find_last_of('/') + 1);
        // Strip a trailing shard suffix "-NNNNN-of-NNNN" and/or ".gguf" to get
        // a search base, then accept any *imatrix*.gguf containing that base.
        std::string base_key = base;
        auto dot = base_key.find(".gguf");
        if (dot != std::string::npos) base_key = base_key.substr(0, dot);
        auto of = base_key.find("-of-");
        if (of != std::string::npos) base_key = base_key.substr(0, of);

        DIR* d = opendir(dir.c_str());
        if (d != nullptr) {
            struct dirent* de = nullptr;
            std::string fallback;
            while ((de = readdir(d)) != nullptr) {
                std::string nm(de->d_name);
                if (nm.find("imatrix") == std::string::npos) continue;
                if (nm.find(".gguf") == std::string::npos) continue;
                std::string full = dir + nm;
                if (full == candidates[0]) continue;
                // Prefer an imatrix whose name contains the quant-specific base
                // (e.g. "Trinity-Mini-Q6_K.imatrix.gguf"), but fall back to any
                // *imatrix*.gguf in the dir (e.g. "Trinity-Mini.imatrix.gguf")
                // since the imatrix is often keyed to the base model name.
                if (!base_key.empty() && nm.find(base_key) != std::string::npos) {
                    candidates.push_back(std::move(full));
                } else if (fallback.empty()) {
                    fallback = std::move(full);
                }
            }
            if (candidates.size() == 1 && !fallback.empty()) {
                candidates.push_back(std::move(fallback));
            }
            closedir(d);
        }
    }

    std::string imatrix_path;
    for (const auto& c : candidates) {
        struct stat st{};
        if (stat(c.c_str(), &st) == 0) { imatrix_path = c; break; }
    }

    if (imatrix_path.empty()) {
        std::cerr << "[Guanaco HerdCache] No imatrix prior found near "
                  << model_path << " (*imatrix*.gguf); cold-start pinning will use runtime stats.\n";
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
    recompute_budgets();
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

        const int topk = (t.budget > 0 ? t.budget : (int)config_.max_active_experts);
        for (int rank = 0; rank < topk && rank < t.num_experts; ++rank) {
            int id = order[rank];
            if (t.pinned[id]) continue;
            if (t.total_hits[id] <= 0) break;  // no calibration usage

            if (!t.loaded[id] && t.mmap_base != nullptr) {
                void* addr = static_cast<char*>(t.mmap_base) + (size_t)id * t.per_expert_bytes;
                madvise(addr, t.per_expert_bytes, MADV_WILLNEED);
                t.loaded[id] = true;
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
              << " expert accesses, distinct experts pinned=" << pinned_total_
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
    const double per_tensor = n_tensors ? (double)pinned_current_ / n_tensors : 0.0;

    int bmin = (int)config_.max_active_experts, bmax = 0;
    for (auto& kv : expert_tensors_) {
        int b = kv.second.budget;
        if (b < bmin) bmin = b;
        if (b > bmax) bmax = b;
    }
    if (n_tensors == 0) { bmin = 0; bmax = 0; }

    std::cerr << "[Guanaco HerdCache] final: pin budget="
              << config_.max_active_experts << " (dynamic per-tensor "
              << bmin << ".." << bmax << ")"
              << ", experts pinned now=" << pinned_current_
              << " (" << per_tensor << " avg / tensor across " << n_tensors
              << " tensors)"
              << ", pin cache=" << pct << "% resident (" << ppct
              << "% hot-pinned)"
               << ", " << total << " expert accesses over "
               << prefetch_calls_ << " prefetch calls"
               << ", evictions=" << evictions_
               << (config_.use_pilot
                   ? (", pilot: " + std::to_string(pilot_calls_) + " calls / " + std::to_string(pilot_slice_reads_) + " slices")
                   : "")
               << (evictions_ > 0
                   ? (", evict_hot=" + std::to_string(evict_hot_) +
                      " evict_reread=" + std::to_string(evict_reread_) +
                      " (" + std::to_string((evict_reread_ * 100) / evictions_) + "% churn)")
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
    if (!enabled_ || expert_ids == nullptr || n <= 0) {
        return;
    }

    ++prefetch_calls_;

    for (auto& kv : expert_tensors_) {
        ExpertTensor& t = kv.second;
        if (t.layer != layer || t.mmap_base == nullptr || t.per_expert_bytes == 0) {
            continue;
        }

        // Classify each router-selected expert for the pin-cache metric.
        //  - pinned: hot expert, always resident, zero page fault
        //  - loaded: warm slice still resident (we have not DONTNEED'd it)
        //  - otherwise: currently evicted; we WILLNEED it back from disk now
        std::unordered_set<int> want;
        for (int i = 0; i < n; ++i) {
            int id = expert_ids[i];
            if (id < 0 || id >= t.num_experts) continue;
            touch(t, id);  // any selected expert is "used now" for LRU
            if (t.pinned[id]) {
                ++pin_hits_;
            } else if (t.loaded[id]) {
                ++warm_hits_;
            } else {
                ++disk_miss_;
                // Tuning instrumentation: was this just evicted recently? If so
                // it's churn (we dropped then re-read the same slice). Scan the
                // small evict log for a matching (tensor,id) within the window.
                for (auto& rec : evict_log_) {
                    if (!rec.used && rec.t == &t && rec.id == id &&
                        eclock_ - rec.clock < kChurnWindow_) {
                        ++evict_reread_;
                        rec.used = true;
                        break;
                    }
                }
                want.insert(id);
            }
        }
        if (want.empty()) {
            continue;
        }

        // Async-resident the selected slices in-place on the mmap. MADV_WILLNEED
        // asks the kernel to page the slice in ahead of the mul_mat_id node;
        // if it is ignored, the later access simply page-faults (correct bytes,
        // slightly later). No copy, no redirect - ggml reads the same address.
        for (int id : want) {
            void* addr = static_cast<char*>(t.mmap_base) + (size_t)id * t.per_expert_bytes;
            madvise(addr, t.per_expert_bytes, MADV_WILLNEED);
            t.loaded[id] = true;
        }
        // Do NOT evict here - the slices we just made resident are needed by
        // the mul_mat_id node that runs immediately after this callback returns.
        // Periodic eviction is handled by maybe_pin_hot_experts()/evict_to_budget.
    }
}

} // namespace guanaco