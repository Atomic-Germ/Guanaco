// Dendritic probe: does a dense model have a sparse, focus-dependent set of
// high-contribution FFN pathways?
//
// This is the gate experiment for pseudo-expert slicing. A dense FFN fires every
// intermediate neuron on every token, so sorting raw activation gives a straight
// line - everything participates. That is expected and uninformative. What
// matters is CONTRIBUTION. The FFN output is
//
//     out_j = sum_i act_i * W_down[j,i]
//
// so pathway i's leverage over the whole output row space is its activation times
// the norm of its down-projection row:
//
//     c_i = act_i * ||W_down[i,:]||_2
//
// We accumulate, per layer, over many tokens:
//
//   perTok@p   share of total contribution carried by that token's own top-p
//              pathways. The DYNAMIC bound: what a router could exploit.
//   global@p   share carried by the single best static top-p set summed over all
//              tokens. The STATIC bound: what importance pruning keeps.
//   overlap    perTok - global. The exploitable headroom. Near zero means one
//              map serves every focus and slicing collapses to pruning. Large
//              means the map is focus-dependent, which is the expensive case.
//
// Also reported: the effective number of participating pathways (participation
// ratio of the contribution distribution), which says whether individual neurons
// are the right unit or something coarser is needed.
//
// Interpretation:
//   perTok@p high AND overlap small  -> knee exists, one global map suffices.
//   perTok@p high AND overlap large  -> knee exists but is focus-dependent.
//   perTok@p near uniform            -> flat curve, slicing is hopeless.
//
// Reads no llama.cpp internals beyond the eval callback; nothing in the model is
// modified. Prints an optional CSV for plotting.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

// Fractions at which to report retained mass.
static const double FRACS[] = {0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50};
static const int    NFRAC = (int) (sizeof(FRACS) / sizeof(FRACS[0]));

struct layer_acc {
    int64_t n_ff     = 0;
    int64_t n_tokens = 0;
    std::vector<float> rownorm;        // n_ff down-projection row norms
    std::vector<float> global;         // per-pathway contribution summed over tokens
    std::vector<double> tot_retain;    // perTok retained mass per FRAC, summed
    std::vector<double> tot_mass{0.0}; // total contribution summed
    double part_ratio_num = 0.0;       // sum c
    double part_ratio_den = 0.0;       // sum c^2
};

struct probe_state {
    std::vector<layer_acc> layers;
    std::vector<std::string> ff_names;
    std::vector<float> scratch;
    std::vector<std::vector<float>> rownorms_shared;  // per-layer down-proj row norms
    std::vector<char> rownorms_done;     // per-layer flag: already extracted
    int cur_domain = 0;

    void ensure(int il) {
        if ((int) layers.size() <= il) {
            layers.resize(il + 1);
            ff_names.resize(il + 1);
        }
    }
};

// One independent scoring pass per domain prompt. Kept separate so the same
// model, loaded once, can be probed on several disparate domains back to back -
// which is the only way to tell a general map from a domain-specific one.
struct domain_acc {
    std::string name;
    std::vector<layer_acc> layers;

    void ensure(int il) {
        if ((int) layers.size() <= il) {
            layers.resize(il + 1);
        }
    }
};

static std::vector<domain_acc> g_domains;
static std::vector<float> g_unit_rownorms;   // all-1.0 fallback leverage

static bool is_layer_of(const char * name, const char * stem, int * il_out) {
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

// The down-projection weight is a graph LEAF and never appears as a named node,
// and no public llama.cpp API hands out weight tensors. Read it from the GGUF
// instead: rows of blk.<il>.ffn_down.weight are pathways, and each row's L2 norm
// is that pathway's leverage over the output row space. Dequantized one row at a
// time so peak memory stays at n_embd floats regardless of model size.
static void load_rownorms_from_gguf(const char * path, probe_state * st) {
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (gctx == nullptr) {
        fprintf(stderr, "[dendrite_probe] cannot open %s as GGUF\n", path);
        return;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[dendrite_probe] cannot reopen %s\n", path);
        gguf_free(gctx);
        return;
    }
    const int n_tensors = gguf_get_n_tensors(gctx);
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
        // gguf_get_tensor_offset is relative to the file start; the caller mapped
        // with no_alloc so the data is not in memory. Read it ourselves.
        const size_t off = gguf_get_tensor_offset(gctx, ti);

        st->ensure(il);
        if ((int) st->rownorms_shared.size() <= il) {
            st->rownorms_shared.resize(il + 1);
            st->rownorms_done.resize(il + 1, 0);
        }
        st->ff_names[il] = name;
        if (st->rownorms_done[(size_t) il]) {
            continue;
        }
        st->rownorms_done[(size_t) il] = 1;
        std::vector<float> & R = st->rownorms_shared[(size_t) il];
        R.assign((size_t) n_ff, 0.0f);

        std::vector<uint8_t> rowbuf(row_bytes);
        std::vector<float> row((size_t) n_embd);
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
                // Isolated Q6_K blocks dequantize to non-finite in the wild; treat
                // those as zero so one bad block cannot poison the ordering.
                if (std::isfinite(row[j])) {
                    acc += (double) row[j] * (double) row[j];
                }
            }
            R[(size_t) i] = (float) std::sqrt(acc);
        }
    }
    close(fd);
    gguf_free(gctx);
}

// SwiGLU activation: act_i for every pathway i of this layer. Scored into the
// current domain's accumulators; row norms are shared across domains because they
// come from the weights, not the input.
static void on_ffn_swiglu(struct ggml_tensor * t, probe_state * st) {
    int il = -1;
    if (!is_layer_of(ggml_get_name(t), "ffn_swiglu", &il)) {
        return;
    }
    if (t->type != GGML_TYPE_F32) {
        return;
    }
    const int64_t n_ff  = t->ne[0];
    const int64_t n_tok = t->ne[1];
    if (n_ff <= 0 || n_tok <= 0) {
        return;
    }
    if (st->cur_domain < 0 || st->cur_domain >= (int) g_domains.size()) {
        return;
    }
    domain_acc & D = g_domains[(size_t) st->cur_domain];
    D.ensure(il);
    layer_acc & L = D.layers[il];

    const std::vector<float> & rn =
        (il < (int) st->rownorms_shared.size() && (int64_t) st->rownorms_shared[(size_t) il].size() == n_ff)
        ? st->rownorms_shared[(size_t) il] : g_unit_rownorms;

    if (L.global.empty()) {
        L.n_ff = n_ff;
        L.global.assign((size_t) n_ff, 0.0f);
        L.tot_retain.assign(NFRAC, 0.0);
    }

    std::vector<float> buf((size_t) n_ff * n_tok);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));

    std::vector<float> c((size_t) n_ff);
    for (int64_t j = 0; j < n_tok; ++j) {
        const float * row = buf.data() + (size_t) j * n_ff;
        double tot = 0.0;
        for (int64_t i = 0; i < n_ff; ++i) {
            const double v = std::fabs((double) row[i]) * (double) rn[(size_t) i];
            c[(size_t) i] = (float) v;
            tot += v;
        }
        if (tot <= 0.0) {
            continue;
        }
        std::sort(c.begin(), c.end(), std::greater<float>());
        for (int f = 0; f < NFRAC; ++f) {
            const int k = std::max(1, (int) (FRACS[f] * n_ff));
            double s = 0.0;
            for (int i = 0; i < k && i < n_ff; ++i) {
                s += c[(size_t) i];
            }
            L.tot_retain[(size_t) f] += s / tot;
        }
        for (int64_t i = 0; i < n_ff; ++i) {
            const float v = std::fabs(row[i]) * rn[(size_t) i];
            L.global[(size_t) i] += v;
            L.part_ratio_num += v;
            L.part_ratio_den += (double) v * (double) v;
        }
        L.tot_mass[0] += tot;
        L.n_tokens++;
    }
}

static long g_n_calls = 0;
static std::string g_first_names;

static bool probe_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (probe_state *) user_data;
    if (ask) {
        return true;
    }
    ++g_n_calls;
    const char * name = ggml_get_name(t);
    if (g_first_names.size() < 400 && strstr(name, "ffn") != nullptr) {
        g_first_names += name;
        g_first_names += ";";
    }
    if (strncmp(name, "ffn_swiglu", 10) == 0) {
        on_ffn_swiglu(t, st);
        return true;
    }
    return true;
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    probe_state st;
    params.cb_eval           = probe_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    // Under streaming Guanaco chains its router hook with this callback instead
    // of replacing it, so the probe sees the same nodes either way.
    load_rownorms_from_gguf(params.model.path.c_str(), &st);
    g_unit_rownorms.assign(st.rownorms_shared.empty() ? 1 : st.rownorms_shared[0].size(), 1.0f);

    // Domains: "# name" header lines in a file. Delimiting on the header, not on
    // blank lines, matters - generated prose has paragraph breaks, and splitting
    // on those silently shatters one domain into many short fragments.
    std::vector<std::pair<std::string, std::string>> domains;  // name, prompt
    if (const char * df = getenv("PROBE_DOMAINS")) {
        FILE * f = fopen(df, "r");
        if (f == nullptr) {
            fprintf(stderr, "[dendrite_probe] cannot open PROBE_DOMAINS=%s\n", df);
            return 1;
        }
        std::string cur_name, cur_txt;
        auto flush = [&]() {
            if (cur_txt.find_first_not_of(" \t\n\r") == std::string::npos) return;
            if (cur_name.empty()) cur_name = "domain" + std::to_string(domains.size() + 1);
            domains.emplace_back(cur_name, cur_txt);
            cur_txt.clear();
        };
        char line[8192];
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '#') {
                flush();
                std::string h(line + 1);
                while (!h.empty() && (h.back() == '\n' || h.back() == '\r')) h.pop_back();
                cur_name = h;
                continue;
            }
            cur_txt += line;
        }
        flush();
        fclose(f);
        fprintf(stderr, "[dendrite_probe] %zu domain(s) from %s\n", domains.size(), df);
    }
    if (domains.empty()) {
        domains.emplace_back("prompt", params.prompt);
    }

    int n_cap = 0;
    if (const char * e = getenv("PROBE_TOKENS")) {
        n_cap = atoi(e);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_batch = params.n_batch > 0 ? (int) params.n_batch : 512;

    for (size_t di = 0; di < domains.size(); ++di) {
        g_domains.resize(di + 1);
        g_domains[di].name = domains[di].first;
        st.cur_domain = (int) di;

        std::vector<llama_token> tokens = common_tokenize(
            ctx, domains[di].second, llama_vocab_get_add_bos(vocab), true);
        if (tokens.empty()) {
            fprintf(stderr, "[dendrite_probe] no tokens for domain '%s'\n", domains[di].first.c_str());
            continue;
        }
        int n_max = (int) tokens.size();
        if (n_cap > 0 && n_cap < n_max) n_max = n_cap;
        fprintf(stderr, "[dendrite_probe] domain %zu '%s': %d tokens\n",
                di, domains[di].first.c_str(), n_max);
        if (n_max < 64) {
            // A short domain still yields numbers, but its map is noise and it
            // quietly drags down every column it appears in.
            fprintf(stderr, "[dendrite_probe] WARNING: domain '%s' has only %d tokens; "
                            "its map is not meaningful and will bias the transfer matrix.\n",
                    domains[di].first.c_str(), n_max);
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        int n_past = 0;
        while (n_past < n_max) {
            const int n = std::min(n_batch, n_max - n_past);
            common_batch batch(ctx);
            for (int i = 0; i < n; ++i) {
                batch.add(tokens[n_past + i], n_past + i, 0, false);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            n_past += n;
        }
    }

    const char * csv = getenv("PROBE_CSV");

    printf("\n=== Dendritic contribution spread ===\n");
    printf("c_i = |act_i| * ||W_down[i,:]||_2   (per-token, per-layer)\n");
    printf("perTok = dynamic bound (that token's own top-p)\n");
    printf("global = static bound (best single top-p set over all tokens)\n");
    printf("overlap = perTok - global = exploitable headroom\n\n");

    const size_t D = g_domains.size();

    // Per-domain summary.
    for (size_t di = 0; di < D; ++di) {
        std::vector<double> sum_per(NFRAC, 0.0), sum_glob(NFRAC, 0.0), sum_eff(1, 0.0);
        int nl = 0;
        for (size_t il = 0; il < g_domains[di].layers.size(); ++il) {
            layer_acc & L = g_domains[di].layers[il];
            if (L.n_tokens == 0 || L.n_ff <= 0) continue;
            const int nff = (int) L.n_ff;

            std::vector<double> per(NFRAC), glob(NFRAC);
            for (int f = 0; f < NFRAC; ++f) per[(size_t) f] = L.tot_retain[(size_t) f] / (double) L.n_tokens;

            double gsum = 0.0;
            for (float v : L.global) gsum += v;
            std::vector<float> g = L.global;
            std::sort(g.begin(), g.end(), std::greater<float>());
            double run = 0.0;
            size_t ri = 0;
            for (int f = 0; f < NFRAC; ++f) {
                const size_t k = std::max<size_t>(1, (size_t) (FRACS[f] * nff));
                while (ri < k && ri < g.size()) run += g[ri++];
                glob[(size_t) f] = gsum > 0.0 ? run / gsum : 0.0;
            }

            double eff = 0.0;
            if (L.part_ratio_den > 0.0 && L.n_tokens > 0) {
                const double T = (double) L.n_tokens;
                eff = (L.part_ratio_num / T) * (L.part_ratio_num / T) / (L.part_ratio_den / T);
                if (eff > (double) nff) eff = (double) nff;
            }
            for (int f = 0; f < NFRAC; ++f) {
                sum_per[(size_t) f] += per[(size_t) f];
                sum_glob[(size_t) f] += glob[(size_t) f];
            }
            sum_eff[0] += eff;
            ++nl;
        }
        if (nl == 0) continue;
        printf("=== domain '%s' (mean over %d layers) ===\n",
               g_domains[di].name.c_str(), nl);
        printf("   %-7s %-9s %-9s %-9s\n", "p", "perTok", "global", "overlap");
        for (int f = 0; f < NFRAC; ++f) {
            printf("   %-7.2f %-9.3f %-9.3f %-9.3f\n",
                   FRACS[f], sum_per[(size_t) f] / nl, sum_glob[(size_t) f] / nl,
                   (sum_per[(size_t) f] - sum_glob[(size_t) f]) / nl);
        }
        printf("   effective participation: %.1f pathways\n\n", sum_eff[0] / nl);
    }

    // Map transfer: build the top-p set from domain A, score it against domain
    // B's mass. Diagonal = perfect specialisation. Off-diagonal near the
    // diagonal = one map serves every focus and there is no per-focus map cost.
    if (D >= 2) {
        printf("=== map transfer: domain A's top-p set scored on domain B ===\n");
        printf("(diagonal is A on A; values near it mean a single map generalises)\n\n");
        for (int f = 0; f < NFRAC; ++f) {
            printf("   p=%-5.2f ", FRACS[f]);
            for (size_t a = 0; a < D; ++a) {
                for (size_t b = 0; b < D; ++b) {
                    double acc = 0.0;
                    int nl = 0;
                    for (size_t il = 0; il < g_domains[a].layers.size(); ++il) {
                        layer_acc & A = g_domains[a].layers[il];
                        if (a == b) {
                            const layer_acc & B = A;
                            if (B.n_tokens == 0 || B.n_ff <= 0) continue;
                            acc += B.tot_retain[(size_t) f] / (double) B.n_tokens;
                            ++nl;
                            continue;
                        }
                        if (il >= g_domains[b].layers.size()) continue;
                        layer_acc & B = g_domains[b].layers[il];
                        if (A.n_ff == 0 || B.n_ff == 0 || A.n_ff != B.n_ff) continue;
                        if (A.global.empty() || B.global.empty()) continue;
                        // rank A's pathways, take top-k, sum B's mass for them
                        std::vector<std::pair<float,int>> rk;
                        rk.reserve(A.global.size());
                        for (size_t i = 0; i < A.global.size(); ++i) {
                            rk.emplace_back(A.global[i], (int) i);
                        }
                        const size_t k = std::min<size_t>(
                            rk.size(), std::max<size_t>(1, (size_t)(FRACS[f] * (int) A.n_ff)));
                        std::partial_sort(rk.begin(), rk.begin() + k, rk.end(),
                            [](const std::pair<float,int> & x, const std::pair<float,int> & y) {
                                return x.first > y.first; });
                        double btot = 0.0, bkeep = 0.0;
                        for (size_t i = 0; i < B.global.size(); ++i) btot += B.global[i];
                        for (size_t i = 0; i < k; ++i) bkeep += B.global[(size_t) rk[i].second];
                        if (btot > 0.0) acc += bkeep / btot;
                        ++nl;
                    }
                    printf("%s:%s=%.3f ", a == b ? "" : "",
                           g_domains[b].name.c_str(), nl > 0 ? acc / nl : 0.0);
                }
            }
            printf("\n");
        }
        printf("\nA row where A-on-B is close to A-on-A means domain A's map works for B.\n");
    }
    return 0;
}