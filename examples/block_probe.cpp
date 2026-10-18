// Is the dendritic sparsity still there at quantisation-block granularity?
//
// Pathway-granular routing is not implementable against a quantised weight:
// gathering scattered individual pathways dequantises a whole block per kept
// pathway, so bytes read amplify by the block size (256x for Q6_K). Routing has
// to select whole blocks.
//
// That may cost most of the sparsity, because a block's value is the sum over
// its 256 pathways. A block holding one strong pathway scores like a block of
// 256 weak ones. This measures the retention curve at block granularity for
// each layer's real quantisation type, next to the pathway-granular curve, so
// the two can be compared directly.
//
// If block retention tracks pathway retention, the design is viable. If it
// collapses toward uniform, the whole pseudo-expert idea dies here, before any
// operator is written.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "dendrite_common.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char * TAG = "block_probe";

struct state {
    dendrite::rownorms rn;
    std::vector<std::vector<float>> global;
    std::vector<char> layer_done;
    std::vector<std::string> nodes_seen;
    std::vector<float> scratch;
    int64_t n_tok_total = 0;
    bool seen_swiglu = false;

    void ensure(int il, int64_t n_ff) {
        if ((int) global.size() <= il) {
            global.resize(il + 1);
            layer_done.resize(il + 1, 0);
        }
        if (layer_done[(size_t) il]) return;
        layer_done[(size_t) il] = 1;
        global[(size_t) il].assign((size_t) n_ff, 0.0f);
    }
};

static void score(state * st, struct ggml_tensor * t, int il) {
    const int64_t n_ff = t->ne[0], n_tok = t->ne[1];
    if (st->rn.n_ff(il) != n_ff) return;
    st->ensure(il, n_ff);
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
    st->n_tok_total += n_tok;
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (state *) ud;
    const char * kind = dendrite::ff_node_kind(ggml_get_name(t));
    if (st->nodes_seen.size() < 40 &&
        std::find(st->nodes_seen.begin(), st->nodes_seen.end(), std::string(ggml_get_name(t))) == st->nodes_seen.end()) {
        st->nodes_seen.emplace_back(ggml_get_name(t));
    }
    if (ask) {
        if (kind == nullptr || t->type != GGML_TYPE_F32) return false;
        if (strcmp(kind, "ffn_out") == 0 && st->seen_swiglu) return false;
        return true;
    }
    if (kind == nullptr || t->type != GGML_TYPE_F32) return true;
    if (strcmp(kind, "ffn_swiglu") == 0) {
        st->seen_swiglu = true;
    } else if (st->seen_swiglu) {
        return true;
    }
    int il = -1;
    if (dendrite::layer_of(ggml_get_name(t), kind, &il)) score(st, t, il);
    return true;
}

// retained mass when keeping the top-fraction of a grouping
static double retention(const std::vector<float> & g, int64_t grp, double p) {
    const int64_t n = (int64_t) g.size();
    const int64_t nb = (n + grp - 1) / grp;
    std::vector<double> bs((size_t) nb, 0.0);
    double tot = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        bs[(size_t) (i / grp)] += g[(size_t) i];
        tot += g[(size_t) i];
    }
    if (tot <= 0.0) {
        return 0.0;
    }
    std::sort(bs.begin(), bs.end(), std::greater<double>());
    const size_t k = std::max<size_t>(1, (size_t) (p * (double) nb));
    double run = 0.0;
    for (size_t r = 0; r < std::min(k, bs.size()); ++r) {
        run += bs[r];
    }
    return run / tot;
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    state st;
    params.cb_eval = cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }

    if (params.load_mode == LLAMA_LOAD_MODE_STREAMING) {
        fprintf(stderr, "[%s] streaming replaces cb_eval; use --load-mode mmap\n", TAG);
        return 2;
    }
    if (!dendrite::load_rownorms(params.model.path.c_str(), st.rn, TAG)) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) { fprintf(stderr, "no tokens; pass -p or -f\n"); return 1; }
    int n_max = (int) tokens.size();
    if (const char * e = getenv("PROBE_TOKENS")) {
        const int v = atoi(e);
        if (v > 0 && v < n_max) n_max = v;
    }
    {
        int n_past = 0;
        while (n_past < n_max) {
            const int n = std::min(512, n_max - n_past);
            common_batch batch(ctx);
            for (int i = 0; i < n; ++i) batch.add(tokens[n_past + i], n_past + i, 0, false);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return 1;
            n_past += n;
        }
    }
    fprintf(stderr, "[%s] scored %lld tokens\n", TAG, (long long) st.n_tok_total);

    int nl = 0;
    for (size_t il = 0; il < st.global.size(); ++il) if (!st.global[il].empty()) ++nl;
    if (nl == 0) {
        fprintf(stderr, "[%s] no FFN nodes matched; saw:\n", TAG);
        for (auto & s : st.nodes_seen) fprintf(stderr, "    %s\n", s.c_str());
        return 1;
    }

    // group sizes worth reporting: 1 (pathway), the real block, and a few
    std::vector<int64_t> grps = {1, 32, 64, 128, 256};
    for (size_t il = 0; il < st.rn.types.size(); ++il) {
        const int64_t b = st.rn.block((int) il);
        if (b > 1 && std::find(grps.begin(), grps.end(), b) == grps.end()) grps.push_back(b);
    }
    std::sort(grps.begin(), grps.end());

    const double fracs[] = {0.05, 0.10, 0.20, 0.35, 0.50, 0.75};
    printf("\n=== Retention at routing granularity (mean over %d layers) ===\n", nl);
    printf("down-proj quant: ");
    {
        std::vector<std::string> ts;
        for (size_t il = 0; il < st.rn.types.size(); ++il) {
            if (st.rn.types[il] == GGML_TYPE_COUNT) continue;
            std::string s = ggml_type_name(st.rn.types[il]);
            if (std::find(ts.begin(), ts.end(), s) == ts.end()) ts.push_back(s);
        }
        for (size_t i = 0; i < ts.size(); ++i) printf("%s%s", i ? ", " : "", ts[i].c_str());
    }
    printf("\nn_ff blocks: ");
    {
        int64_t nff = 0;
        for (size_t il = 0; il < st.global.size(); ++il) if (!st.global[il].empty()) { nff = (int64_t) st.global[il].size(); break; }
        for (size_t i = 0; i < grps.size(); ++i) printf("%s%lld", i ? " / " : "", (long long) ((nff + grps[i] - 1) / grps[i]));
    }
    printf("\n\n");
    printf("%8s", "p");
    for (int64_t g : grps) printf(" grp=%-6lld", (long long) g);
    printf("\n");

    for (double p : fracs) {
        printf("%8.2f", p);
        for (int64_t grp : grps) {
            double sum = 0.0;
            int cnt = 0;
            for (size_t il = 0; il < st.global.size(); ++il) {
                if (st.global[il].empty()) continue;
                sum += retention(st.global[il], grp, p);
                ++cnt;
            }
            printf(" %9.3f ", cnt ? sum / cnt : 0.0);
        }
        printf("\n");
    }
    printf("\ngrp=1 is the pathway-granular curve the dense design assumed.\n"
           "The real quantisation block is what routing must use; if its curve\n"
           "collapses toward uniform, block-granular routing is not viable.\n");

    llama_backend_free();
    return 0;
}
