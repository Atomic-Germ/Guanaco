// guanaco-repack: prototype sidecar generator.
//
// For every fused MoE expert tensor in a GGUF, read each expert's raw bytes
// and repack them into the CPU backend's blocked layout (the same transform
// ggml applies at load time via CPU_REPACK), then append the repacked bytes
// to a sidecar file next to the GGUF. A small JSON manifest records, per
// tensor, the expert count, per-expert repacked stride, and the byte offset
// of each expert slice in the sidecar.
//
// This lets the runtime skip CPU_REPACK entirely and stream already-repacked
// expert slices directly into the slab.
//
// Prototype: validates the concept (correct repacked stride, identical layout
// to what ggml would produce at load). Not yet wired into the runtime.

#include "gguf.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>

static size_t file_size(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n > 0 ? (size_t)n : 0;
}

static bool read_at(int fd, size_t off, void* dst, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(fd, (char*)dst + done, len - done, (off_t)(off + done));
        if (r <= 0) return false;
        done += (size_t)r;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [out.sidecar]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    std::string sidecar_path = model_path + ".guanaco_repack";
    if (argc >= 3) sidecar_path = argv[2];

    gguf_init_params params{/*.no_alloc =*/false, /*.ctx_override =*/nullptr};
    struct gguf_context* ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "failed to open GGUF: %s\n", model_path.c_str());
        return 1;
    }

    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open for reading: %s\n", model_path.c_str());
        gguf_free(ctx);
        return 1;
    }

    FILE* out = fopen(sidecar_path.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "failed to open sidecar for writing: %s\n", sidecar_path.c_str());
        close(fd);
        gguf_free(ctx);
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    int redirected = 0;
    int skipped = 0;

    // Manifest (simple text lines): name|num_experts|per_expert_bytes|sidecar_offset
    std::string manifest;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr) continue;
        if (strstr(name, "exps.weight") == nullptr &&
            strstr(name, ".experts.") == nullptr) {
            continue;
        }

        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        if (type == GGML_TYPE_MXFP4) {
            fprintf(stderr, "skip (MXFP4 unsupported): %s\n", name);
            ++skipped;
            continue;
        }

        const int64_t* ne = gguf_get_tensor_ne(ctx, i);
        int num_experts = 1;
        for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
            if (ne[d] > 1) { num_experts = (int)ne[d]; break; }
        }
        if (num_experts <= 1) {
            ++skipped;
            continue;
        }

        const size_t tensor_size = gguf_get_tensor_size(ctx, i);
        const size_t file_offset = gguf_get_tensor_offset(ctx, i);
        if (tensor_size % (size_t)num_experts != 0) {
            fprintf(stderr, "skip (stride not even): %s\n", name);
            ++skipped;
            continue;
        }
        const size_t raw_stride = tensor_size / (size_t)num_experts;

        // Build a one-expert tensor descriptor: collapse the expert (trailing
        // ne[d]>1) dim to 1. ggml picks the repack variant from ne[1] % block
        // and CPU features, so a single expert must keep the same ne[0]/ne[1]
        // as the full tensor to get the identical layout.
        int expert_dim = 0;
        for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
            if (ne[d] > 1) { expert_dim = d; break; }
        }
        struct ggml_init_params cp = { .mem_size = 1ull << 20, .mem_buffer = nullptr, .no_alloc = true };
        struct ggml_context* gctx = ggml_init(cp);
        if (!gctx) { fprintf(stderr, "ggml_init failed\n"); continue; }
        std::vector<int64_t> ene(GGML_MAX_DIMS, 1);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) ene[d] = ne[d];
        ene[expert_dim] = 1; // collapse expert dim
        struct ggml_tensor* t = ggml_new_tensor(gctx, type, GGML_MAX_DIMS, ene.data());
        if (!t) { ggml_free(gctx); continue; }

        const size_t repacked_stride = ggml_nbytes(t);
        if (repacked_stride != raw_stride) {
            // repack preserves total size and outer stride, so per-expert stride
            // should match; if not, abort rather than write a wrong sidecar.
            fprintf(stderr, "stride mismatch for %s: raw=%zu repacked=%zu (num_experts=%d) - aborting\n",
                    name, raw_stride, repacked_stride, num_experts);
            ggml_free(gctx);
            fclose(out);
            remove(sidecar_path.c_str());
            gguf_free(ctx);
            close(fd);
            return 2;
        }

        std::vector<uint8_t> raw(raw_stride);
        std::vector<uint8_t> repacked(repacked_stride);

        // If ggml has no repack variant for this type on this CPU (e.g. Q8_0
        // on x86), the "repacked" layout IS the raw layout: copy as-is.
        const int repack_rc = ggml_backend_cpu_repack_tensor(t, raw.data(), repacked.data());
        const bool needs_repack = (repack_rc == 0);
        fprintf(stderr, "  [dbg] %s type=%s expert_dim_collapsed ne=[%lld,%lld,%lld,%lld] repack_rc=%d neon=%d\n",
                name, ggml_type_name(type),
                (long long)ene[0],(long long)ene[1],(long long)ene[2],(long long)ene[3],
                repack_rc, (int)ggml_cpu_has_neon());

        size_t sidecar_offset = (size_t)ftell(out);
        bool ok = true;
        for (int e = 0; e < num_experts; ++e) {
            if (!read_at(fd, file_offset + (size_t)e * raw_stride, raw.data(), raw_stride)) {
                fprintf(stderr, "read failed for %s expert %d\n", name, e);
                ok = false;
                break;
            }
            const void* out_bytes = raw.data();
            if (needs_repack) {
                if (ggml_backend_cpu_repack_tensor(t, raw.data(), repacked.data()) != 0) {
                    fprintf(stderr, "repack failed for %s expert %d (type=%s)\n",
                            name, e, ggml_type_name(type));
                    ok = false;
                    break;
                }
                out_bytes = repacked.data();
            }
            if (fwrite(out_bytes, 1, repacked_stride, out) != repacked_stride) {
                fprintf(stderr, "write failed for %s expert %d\n", name, e);
                ok = false;
                break;
            }
        }

        ggml_free(gctx);

        if (!ok) {
            fclose(out);
            remove(sidecar_path.c_str());
            gguf_free(ctx);
            close(fd);
            return 2;
        }

        char line[1024];
        snprintf(line, sizeof(line), "%s|%d|%zu|%zu\n",
                 name, num_experts, repacked_stride, sidecar_offset);
        manifest += line;
        fprintf(stderr, "repacked %s: %d experts x %zu bytes @ sidecar offset %zu\n",
                name, num_experts, repacked_stride, sidecar_offset);
        ++redirected;
    }

    // Manifest trailer: a sentinel line, then the JSON-ish block is just the
    // text lines above. We append a marker so the runtime can find it.
    const char* trailer = "GUNACO_REPACK_MANIFEST_END\n";
    fwrite(trailer, 1, strlen(trailer), out);
    fwrite(manifest.data(), 1, manifest.size(), out);

    fclose(out);
    gguf_free(ctx);
    close(fd);

    fprintf(stderr, "wrote sidecar %s: %d tensors repacked, %d skipped\n",
            sidecar_path.c_str(), redirected, skipped);
    return 0;
}
