// Decisive test for the dendritic map: does retained contribution mass predict
// retained model quality?
//
// dendrite_probe answers a bookkeeping question - how much of a layer's FFN
// output magnitude a top-p pathway set carries. That is an L2 statement about
// perturbation size, and it is exact as such. What it does not establish is
// that a small perturbation means a good model. That is the assumption the
// whole pseudo-expert idea rests on, and this tool is what tests it.
//
// Method: derive a map on one text, then mask the pathways it does not keep
// and measure negative log likelihood on held-out text. If the NLL curve tracks
// the predicted retained mass, the map is a real lever. If quality collapses
// while predicted mass stays high, c_i is a proxy artifact and no amount of map
// engineering helps.
//
// The mask is applied by zeroing ffn_swiglu channels in the eval callback. That
// works because ggml_backend_sched computes nodes in graph order and invokes
// the callback with ask=false after node t computes and the backend has
// synchronized, but before any consumer of t runs. Zeroing channel i removes
// exactly act_i * W_down[:,i] from the layer output.
//
// Requires CPU offload (-ngl 0): the tensor data must be host-writable. On a
// device backend t->data is device memory and the in-place write would fault.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "dendrite_common.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * TAG = "dendrite_ablate";

struct ablate_state {
    dendrite::rownorms rn;

    // map being evaluated: for each layer, a keep-mask over pathways
    std::vector<std::vector<char>> keep;
    bool masking = false;
    // dynamic: each token keeps its own top-p. This is what routing would
    // actually do; the static keep-mask above is the pruning control.
    bool dynamic = false;
    // block-granular: routing can only select whole quantisation blocks, since
    // gathering scattered pathways out of a quantised weight dequantises a
    // whole block per kept pathway. 0 means pathway-granular.
    int64_t grp = 0;
    double frac = 0.0;

    // calibration accumulation
    bool                calibrating = false;
    std::vector<std::vector<float>> global;
    int64_t              n_calib_tokens = 0;

    std::vector<float> scratch;
    long n_masked = 0;
    bool seen_swiglu = false;
    std::vector<std::string> nodes_seen;
    std::vector<char>        layer_done;

    void ensure_layer(int il, int64_t n_ff) {
        if ((int) global.size() <= il) {
            global.resize(il + 1);
            layer_done.resize(il + 1, 0);
        }
        if (layer_done[(size_t) il]) {
            return;
        }
        layer_done[(size_t) il] = 1;
        global[(size_t) il].assign((size_t) n_ff, 0.0f);
    }
};

// Contributions per pathway: c_i = |act_i| * ||W_down[i,:]||_2.
static void score_layer(ablate_state * st, struct ggml_tensor * t, int il) {
    const int64_t n_ff  = t->ne[0];
    const int64_t n_tok = t->ne[1];
    const int64_t rn_n = st->rn.n_ff(il);
    if (rn_n != n_ff) {
        fprintf(stderr, "[%s] layer %d width mismatch: ff node %lld, down-proj %lld; skipping\n",
                TAG, il, (long long) n_ff, (long long) rn_n);
        return;
    }
    st->ensure_layer(il, n_ff);

    st->scratch.resize((size_t) n_ff * (size_t) n_tok);
    ggml_backend_tensor_get(t, st->scratch.data(), 0, st->scratch.size() * sizeof(float));

    std::vector<float> & g = st->global[(size_t) il];
    const std::vector<float> & R = st->rn.per_layer[(size_t) il];
    for (int64_t j = 0; j < n_tok; ++j) {
        const float * row = st->scratch.data() + (size_t) j * n_ff;
        for (int64_t i = 0; i < n_ff; ++i) {
            g[(size_t) i] += std::fabs((double) row[i]) * (double) R[(size_t) i];
        }
    }
    st->n_calib_tokens += n_tok;
}

static bool ablate_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (ablate_state *) user_data;
    const char * kind = dendrite::ff_node_kind(ggml_get_name(t));

    if (st->nodes_seen.size() < 40 &&
        std::find(st->nodes_seen.begin(), st->nodes_seen.end(), std::string(ggml_get_name(t))) == st->nodes_seen.end()) {
        st->nodes_seen.emplace_back(ggml_get_name(t));
    }

    if (ask) {
        // Must claim every FFN node in both passes. Returning false lets the
        // scheduler batch nodes together, and it only reports the last node of
        // each batch - so a calibration pass that claims nothing sees no FFN
        // node at all. Claiming them isolates each one and forces a sync.
        if (kind == nullptr || t->type != GGML_TYPE_F32) {
            return false;
        }
        if (strcmp(kind, "ffn_out") == 0 && st->seen_swiglu) {
            // ffn_out is the collapsed FFN result, width n_embd. It carries no
            // per-pathway axis, so it is only a fallback for builds that never
            // emit ffn_swiglu at all.
            return false;
        }
        return true;
    }
    if (kind == nullptr || t->type != GGML_TYPE_F32) {
        return true;
    }
    if (strcmp(kind, "ffn_swiglu") == 0) {
        st->seen_swiglu = true;
    } else if (st->seen_swiglu) {
        return true;
    }

    int il = -1;
    if (!dendrite::layer_of(ggml_get_name(t), kind, &il)) {
        return true;
    }

    if (st->calibrating) {
        score_layer(st, t, il);
        return true;
    }

    if (!st->masking || il >= (int) st->keep.size()) {
        return true;
    }
    const int64_t n_ff  = t->ne[0];
    const int64_t n_tok = t->ne[1];

    float * data = (float *) t->data;
    if (data == nullptr) {
        fprintf(stderr, "[%s] layer %d: null tensor data; run with -ngl 0 -dev none\n", TAG, il);
        st->masking = false;
        return true;
    }
    long zeroed = 0;

    if (st->dynamic) {
        // Rank every token's own pathways by c_i and keep that token's top-p.
        // nth_element is O(n) where partial_sort is O(n log n); at 64 layers and
        // hundreds of tokens the difference is the whole runtime.
        const int64_t rn_n = st->rn.n_ff(il);
        if (rn_n != n_ff) {
            return true;
        }
        const std::vector<float> & R = st->rn.per_layer[(size_t) il];
        const int64_t grp = st->grp > 0 ? st->grp : 1;
        const int64_t nb  = (n_ff + grp - 1) / grp;
        const size_t k = std::max<size_t>(1, (size_t) (st->frac * (double) nb));
        std::vector<std::pair<double,int32_t>> bidx((size_t) nb);
        std::vector<char> keepblk((size_t) nb, 0);
        for (int64_t j = 0; j < n_tok; ++j) {
            float * row = data + (size_t) j * n_ff;
            for (int64_t b = 0; b < nb; ++b) {
                double s = 0.0;
                const int64_t e = std::min(n_ff, (b + 1) * grp);
                for (int64_t i = b * grp; i < e; ++i) {
                    s += std::fabs((double) row[i]) * (double) R[(size_t) i];
                }
                bidx[(size_t) b] = { s, (int32_t) b };
            }
            std::nth_element(bidx.begin(), bidx.begin() + std::min(k, bidx.size()) - 1, bidx.end(),
                [](const std::pair<double,int32_t> & a, const std::pair<double,int32_t> & b) {
                    return a.first > b.first; });
            std::fill(keepblk.begin(), keepblk.end(), (char) 0);
            const size_t keep_n = std::min(k, bidx.size());
            for (size_t r = 0; r < keep_n; ++r) {
                keepblk[(size_t) bidx[r].second] = 1;
            }
            for (int64_t b = 0; b < nb; ++b) {
                if (keepblk[(size_t) b]) continue;
                const int64_t e = std::min(n_ff, (b + 1) * grp);
                for (int64_t i = b * grp; i < e; ++i) row[i] = 0.0f;
            }
            zeroed += (long) (n_ff - (int64_t) (keep_n * grp));
        }
        st->n_masked += zeroed;
        return true;
    }

    const std::vector<char> & keep = st->keep[(size_t) il];
    if (keep.empty() || (int64_t) keep.size() != n_ff) {
        return true;
    }

    // Zero the dropped pathways in place. Safe: the allocator cannot hand this
    // memory to another tensor while ffn_swiglu is still live, and ffn_out has
    // not run yet.
    for (int64_t j = 0; j < n_tok; ++j) {
        float * row = data + (size_t) j * n_ff;
        for (int64_t i = 0; i < n_ff; ++i) {
            if (!keep[(size_t) i]) {
                row[i] = 0.0f;
                ++zeroed;
            }
        }
    }
    st->n_masked += zeroed;
    return true;
}

// Mean NLL over tokens [n_skip, n_tokens). Lower is better.
static bool eval_nll(struct llama_context * ctx, const std::vector<llama_token> & tokens,
                     int n_skip, double * nll_out, int * n_out) {
    const int n = (int) tokens.size();
    const int n_batch = 512;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    double sum = 0.0;
    int count = 0;
    int n_past = 0;
    while (n_past < n) {
        const int nb = std::min(n_batch, n - n_past);
        common_batch batch(ctx);
        for (int i = 0; i < nb; ++i) {
            batch.add(tokens[n_past + i], n_past + i, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "[%s] decode failed\n", TAG);
            return false;
        }
        const float * logits = llama_get_logits(ctx);
        if (logits == nullptr) {
            fprintf(stderr, "[%s] no logits\n", TAG);
            return false;
        }
        for (int i = 0; i < nb; ++i) {
            const int pos = n_past + i;
            if (pos < n_skip || pos + 1 >= n) {
                continue;  // no target, and the last token has no successor
            }
            const float * row = logits + (size_t) i * n_vocab;
            float maxl = row[0];
            for (int v = 1; v < n_vocab; ++v) maxl = std::max(maxl, row[v]);
            double se = 0.0;
            for (int v = 0; v < n_vocab; ++v) se += std::exp((double) row[v] - maxl);
            const double lp = (double) row[tokens[pos + 1]] - maxl - std::log(se);
            sum += -lp;
            ++count;
        }
        n_past += nb;
    }
    if (count == 0) {
        return false;
    }
    *nll_out = sum / count;
    *n_out = count;
    return true;
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ablate_state st;
    params.cb_eval           = ablate_eval;
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

    if (params.load_mode == LLAMA_LOAD_MODE_STREAMING) {
        fprintf(stderr, "[%s] masking needs a plain host buffer; use --load-mode mmap -ngl 0\n", TAG);
        return 2;
    }
    if (params.n_gpu_layers > 0) {
        fprintf(stderr, "[%s] this tool zeroes tensor data in place, which is device memory "
                        "under -ngl > 0. Re-run with -ngl 0.\n", TAG);
        return 2;
    }
    // Zeroing in place also means synchronizing after every FFN node, which no
    // normal llama.cpp path does. With a GPU backend in the scheduler that
    // trips a command-pool lifetime bug and crashes during graph cleanup, so
    // keep every node on the CPU backend.
    params.no_op_offload = true;

    if (!dendrite::load_rownorms(params.model.path.c_str(), st.rn, TAG)) {
        return 1;
    }

    // Two texts: -p derives the map, -f scores it. Using one text for both
    // would measure memorisation, not generalisation.
    const char * calib_path = getenv("ABLATE_CALIB");
    if (calib_path == nullptr) {
        fprintf(stderr, "[%s] set ABLATE_CALIB=<file> to the derivation text; -f is the held-out text\n", TAG);
        return 1;
    }
    const auto read_file = [](const char * path, std::string & out) {
        FILE * f = fopen(path, "r");
        if (f == nullptr) return false;
        char buf[8192];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
        fclose(f);
        return true;
    };
    std::string calib_txt, eval_txt = params.prompt;
    if (!read_file(calib_path, calib_txt) || eval_txt.empty()) {
        fprintf(stderr, "[%s] need readable ABLATE_CALIB and a non-empty -f/-p\n", TAG);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const auto tokens_of = [&](const std::string & s) {
        return common_tokenize(ctx, s, add_bos, true);
    };
    std::vector<llama_token> calib_tokens = tokens_of(calib_txt);
    std::vector<llama_token> eval_tokens  = tokens_of(eval_txt);
    if (calib_tokens.empty() || eval_tokens.empty()) {
        fprintf(stderr, "[%s] empty tokenization\n", TAG);
        return 1;
    }

    int n_calib = (int) calib_tokens.size();
    if (const char * e = getenv("ABLATE_TOKENS")) {
        const int v = atoi(e);
        if (v > 0 && v < n_calib) n_calib = v;
    }
    fprintf(stderr, "[%s] deriving map on %d tokens, scoring %d held-out tokens\n",
            TAG, n_calib, (int) eval_tokens.size());

    // Pass 1: derive the map.
    st.calibrating = true;
    {
        int n_past = 0;
        const int n_batch = 512;
        while (n_past < n_calib) {
            const int n = std::min(n_batch, n_calib - n_past);
            common_batch batch(ctx);
            for (int i = 0; i < n; ++i) batch.add(calib_tokens[n_past + i], n_past + i, 0, false);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                fprintf(stderr, "[%s] calibration decode failed\n", TAG);
                return 1;
            }
            n_past += n;
        }
    }
    st.calibrating = false;

    int layers_scored = 0;
    for (size_t il = 0; il < st.global.size(); ++il) {
        if (!st.global[il].empty()) ++layers_scored;
    }
    if (layers_scored == 0) {
        fprintf(stderr, "[%s] ERROR: no FFN nodes matched. Saw %zu node names, e.g.:\n",
                TAG, st.nodes_seen.size());
        for (size_t i = 0; i < st.nodes_seen.size() && i < 15; ++i) {
            fprintf(stderr, "    %s\n", st.nodes_seen[i].c_str());
        }
        return 1;
    }
    fprintf(stderr, "[%s] map derived over %d layers, %lld tokens\n",
            TAG, layers_scored, (long long) st.n_calib_tokens);

    // Pass 2: mask and score. Baseline first so every row has a dNLL.
    const double fracs[] = { 1.00, 0.75, 0.50, 0.35, 0.20, 0.10, 0.05, 0.02 };
    double base_nll = 0.0;
    int    base_cnt = 0;

    st.grp = 0;
    if (const char * b = getenv("ABLATE_GRP")) {
        const int64_t v = atoll(b);
        if (v > 1) st.grp = v;
    }

    printf("\n=== Dendritic mask ablation (held-out NLL) ===\n");
    printf("calib: %s\n", calib_path);
    if (st.grp > 0) {
        printf("routing granularity: %lld pathways (block-granular)\n", (long long) st.grp);
    }
    printf("%6s %12s %12s %10s\n", "p", "predicted", "NLL", "dNLL");
    for (double p : fracs) {
        st.keep.assign(st.global.size(), {});
        double predicted = 0.0;
        int    npred = 0;
        for (size_t il = 0; il < st.global.size(); ++il) {
            if (st.global[il].empty()) continue;
            const std::vector<float> & g = st.global[il];
            const int64_t grp = st.grp > 0 ? st.grp : 1;
            const int64_t nb  = ((int64_t) g.size() + grp - 1) / grp;
            std::vector<char> keep(g.size(), 0);
            double tot = 0.0;
            for (float v : g) tot += v;
            if (tot <= 0.0) continue;
            std::vector<std::pair<double,int32_t>> bs((size_t) nb);
            for (int64_t b = 0; b < nb; ++b) {
                double s = 0.0;
                for (int64_t i = b * grp; i < std::min((int64_t) g.size(), (b + 1) * grp); ++i) {
                    s += g[(size_t) i];
                }
                bs[(size_t) b] = { s, (int32_t) b };
            }
            const size_t k = std::max<size_t>(1, (size_t) (p * (double) nb));
            std::partial_sort(bs.begin(), bs.begin() + std::min(k, bs.size()), bs.end(),
                [](const std::pair<double,int32_t> & a, const std::pair<double,int32_t> & b) {
                    return a.first > b.first; });
            double kept = 0.0;
            for (size_t r = 0; r < std::min(k, bs.size()); ++r) {
                const int64_t lo = bs[r].second * grp;
                const int64_t hi = std::min((int64_t) g.size(), (lo + grp));
                for (int64_t i = lo; i < hi; ++i) { keep[(size_t) i] = 1; kept += g[(size_t) i]; }
            }
            st.keep[il] = keep;
            predicted += kept / tot;
            ++npred;
        }
        if (npred == 0) continue;
        predicted /= npred;

        st.masking = (p < 1.0);
        st.dynamic = st.masking && getenv("ABLATE_DYNAMIC") != nullptr;
        st.frac    = p;
        st.n_masked = 0;
        llama_memory_clear(llama_get_memory(ctx), true);

        double nll = 0.0;
        int cnt = 0;
        if (!eval_nll(ctx, eval_tokens, 0, &nll, &cnt)) {
            fprintf(stderr, "[%s] eval failed at p=%.2f\n", TAG, p);
            return 1;
        }
        if (p >= 1.0) {
            base_nll = nll;
            base_cnt = cnt;
        }
        char dbuf[32];
        if (base_cnt) {
            snprintf(dbuf, sizeof(dbuf), "%+.4f", nll - base_nll);
        } else {
            snprintf(dbuf, sizeof(dbuf), "%s", "-");
        }
        printf("%6.2f %12.3f %12.4f %10s%s\n", p, predicted, nll, dbuf,
               st.dynamic ? "  (dynamic)" : (st.masking ? "  (static)" : ""));
        fflush(stdout);
    }

    printf("\nIf the map is real, dNLL stays near zero while predicted stays high.\n"
           "If dNLL climbs while predicted holds, c_i is a proxy for magnitude\n"
           "but not for quality.\n");

    llama_backend_free();
    return 0;
}
