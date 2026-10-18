// Shared plumbing for the dense pseudo-expert (dendritic) tools.
//
// The down-projection row norms live here because no public API exposes a graph
// leaf: W_down is a weight, so the only way to read it is the GGUF on disk,
// one dequantized row at a time. Both dendrite_probe (scoring) and
// dendrite_ablate (masking) need them, and a second copy would drift.

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace dendrite {

// Per-layer row norms of ffn_down, indexed by layer.
struct rownorms {
    std::vector<std::vector<float>> per_layer;
    std::vector<std::string>        names;
    std::vector<ggml_type>          types;

    int64_t n_ff(int il) const {
        return il < (int) per_layer.size() ? (int64_t) per_layer[(size_t) il].size() : 0;
    }

    // Pathways per quantisation block. Routing must be block-granular: gathering
    // scattered individual pathways out of a quantised weight dequantises a whole
    // block to use one element, which amplifies bytes read by the block size.
    int64_t block(int il) const {
        if (il >= (int) types.size()) {
            return 256;
        }
        const ggml_type_traits * tt = ggml_get_type_traits(types[(size_t) il]);
        return tt != nullptr && tt->blck_size > 0 ? tt->blck_size : 256;
    }
};

// true for "ffn_swiglu-<il>" / "ffn_out-<il>" style names, writing the layer.
inline bool layer_of(const char * name, const char * stem, int * il_out) {
    if (strncmp(name, stem, strlen(stem)) != 0) {
        return false;
    }
    const char * dash = strrchr(name, '-');
    if (dash == nullptr) {
        return false;
    }
    const int il = atoi(dash + 1);
    if (il < 0) {
        return false;
    }
    *il_out = il;
    return true;
}

// Which FFN node carries per-pathway signal for this build? Prefers the fused
// SwiGLU activation, falls back to the FFN result. Returns nullptr for others.
inline const char * ff_node_kind(const char * name) {
    if (strncmp(name, "ffn_swiglu", 10) == 0) return "ffn_swiglu";
    if (strncmp(name, "ffn_out",    7) == 0) return "ffn_out";
    return nullptr;
}

// Read ||W_down[i,:]||_2 for every pathway i of every layer, straight from the
// GGUF. Returns false if the file could not be read as a GGUF at all.
inline bool load_rownorms(const char * path, rownorms & out, const char * tag) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (gctx == nullptr) {
        fprintf(stderr, "[%s] cannot open %s as GGUF\n", tag, path);
        return false;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[%s] cannot reopen %s\n", tag, path);
        gguf_free(gctx);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gctx);
    std::vector<char> done;
    for (int ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(gctx, ti);
        const char * tail = strstr(name, ".ffn_down.weight");
        if (tail == nullptr || name == tail) {
            continue;
        }
        const int il = atoi(name + 4);  // "blk.<il>.ffn_down.weight"
        if (il < 0) {
            continue;
        }
        const enum ggml_type type = gguf_get_tensor_type(gctx, ti);
        const int64_t * ne = gguf_get_tensor_ne(gctx, ti);
        // ggml stores matmul operands as [n_in, n_out], so the down projection
        // reads [n_ff, n_embd]: ne[0] is the pathway count, ne[1] the model
        // width. Reading these the other way round yields n_ff == n_embd-scaled
        // garbage and every layer silently mismatches against ffn_swiglu.
        const int64_t n_ff   = ne[0];
        const int64_t n_embd = ne[1];
        if (n_embd <= 0 || n_ff <= 0) {
            continue;
        }
        const size_t row_bytes = ggml_row_size(type, n_embd);
        // gguf_get_tensor_offset is relative to the file start; no_alloc was set
        // so the data is not in memory. Read it ourselves.
        const size_t off = gguf_get_tensor_offset(gctx, ti);

        if ((int) out.per_layer.size() <= il) {
            out.per_layer.resize(il + 1);
            out.names.resize(il + 1);
            out.types.resize(il + 1, GGML_TYPE_COUNT);
            done.resize(il + 1, 0);
        }
        out.names[(size_t) il] = name;
        out.types[(size_t) il] = type;
        if (done[(size_t) il]) {
            continue;
        }
        done[(size_t) il] = 1;

        std::vector<float> & R = out.per_layer[(size_t) il];
        R.assign((size_t) n_ff, 0.0f);

        std::vector<uint8_t> rowbuf(row_bytes);
        std::vector<float>   row((size_t) n_embd);
        for (int64_t i = 0; i < n_ff; ++i) {
            const off_t at = (off_t) off + (off_t) i * (off_t) row_bytes;
            if (pread(fd, rowbuf.data(), row_bytes, at) != (ssize_t) row_bytes) {
                continue;
            }
            if (type == GGML_TYPE_F32) {
                memcpy(row.data(), rowbuf.data(), row_bytes);
            } else if (type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) rowbuf.data(), row.data(), (int) n_embd);
            } else {
                const auto * tt = ggml_get_type_traits(type);
                if (tt->to_float == nullptr) {
                    continue;
                }
                tt->to_float(rowbuf.data(), row.data(), (int) n_embd);
            }
            double acc = 0.0;
            for (int64_t j = 0; j < n_embd; ++j) {
                // Isolated Q6_K blocks dequantize to non-finite in the wild;
                // treat those as zero so one bad block cannot poison the
                // ordering. We only need the ranking to be sane.
                if (std::isfinite(row[(size_t) j])) {
                    acc += (double) row[(size_t) j] * (double) row[(size_t) j];
                }
            }
            R[(size_t) i] = (float) std::sqrt(acc);
        }
    }
    close(fd);
    gguf_free(gctx);
    return true;
}

// Indices of the top-p pathways by score, descending. O(n log n) per layer.
inline std::vector<int32_t> top_p(const std::vector<float> & score, double p) {
    std::vector<int32_t> idx(score.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = (int32_t) i;
    }
    const size_t k = std::max<size_t>(1, (size_t) (p * (double) score.size()));
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, idx.size()), idx.end(),
        [&](int32_t a, int32_t b) { return score[(size_t) a] > score[(size_t) b]; });
    idx.resize(std::min(k, idx.size()));
    return idx;
}

}  // namespace dendrite
