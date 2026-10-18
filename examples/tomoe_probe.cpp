// Phase-0 measurement probe for the ToMoE-style dense pseudo-expert experiment.
//
// It runs a dense model and captures the FFN activation tensor for each layer
// (length n_ff per token). Two node names carry that signal depending on how
// the architecture builds its FFN:
//
//   ffn_swiglu-<il>  fused SwiGLU activation (gated SiLU * up)
//   ffn_out-<il>     the FFN result itself, i.e. pathway i's actual
//                    contribution act_i * W_down[:,i] summed over the output
//
// ffn_out is the better signal when present - it is contribution, not
// activation - but not every build emits it, so we take whichever appears and
// record which. For each layer it buckets the
// intermediate channels into E uniform pseudo-experts and records the per-token
// energy (sum of squares) per expert. After the run it reports, per layer:
//
//   - oracle top-k mass: the average per-token share of FFN activation energy
//     captured by that token's own top-k experts.
//   - global top-k mass: the share captured by the single global top-k set
//     (the static control, what importance pruning would keep).
//   - the gain of dynamic over static routing.
//
// If the oracle mass is close to the global mass, per-token routing has nothing
// to exploit and the experiment should stop. If the oracle mass is much higher,
// dynamic pseudo-expert routing is worth building.
//
// This probe changes nothing in llama.cpp and is not part of production.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct probe_data {
    int E    = 17;
    int topk = 4;

    int64_t n_ff     = 0;
    std::string swiglu_name_seen;
    std::vector<char> layer_taken;       // one capture per layer, whichever node won
    std::vector<std::string> nodes_seen; // diagnostic when nothing matches

    // tokE[layer] holds ntokens[layer] * E energies, token-major.
    std::vector<std::vector<float>> tokE;
    std::vector<int64_t>            ntokens;

    void ensure(int il, int64_t n_ff_expected) {
        if ((int) tokE.size() <= il) {
            tokE.resize(il + 1);
            ntokens.resize(il + 1, 0);
            layer_taken.resize(il + 1, 0);
        }
        (void) n_ff_expected;
    }
};

// Which FFN node are we scoring? Prefer the fused activation, fall back to the
// FFN output. Returns nullptr for any other tensor.
static const char * ff_node_kind(const char * name) {
    if (strncmp(name, "ffn_swiglu", 10) == 0) {
        return "ffn_swiglu";
    }
    if (strncmp(name, "ffn_out",    7) == 0) {
        return "ffn_out";
    }
    return nullptr;
}

static bool probe_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (probe_data *) user_data;

    if (ask) {
        return true;
    }

    const char * kind = ff_node_kind(t->name);
    if (kind == nullptr) {
        // Keep a short sample of node names purely so that "captured nothing"
        // can explain itself instead of printing an empty table.
        if (d->nodes_seen.size() < 40 &&
            std::find(d->nodes_seen.begin(), d->nodes_seen.end(), std::string(t->name)) == d->nodes_seen.end()) {
            d->nodes_seen.emplace_back(t->name);
        }
        return true;
    }

    const char * dash = strrchr(t->name, '-');
    if (dash == nullptr) {
        return true;
    }
    const int il = atoi(dash + 1);
    if (il < 0) {
        return true;
    }

    d->ensure(il, 0);
    // Both node names can exist for one layer. Score the layer once, from the
    // preferred node, so tokens are not double-counted.
    if (d->layer_taken[il]) {
        if (d->swiglu_name_seen.find("ffn_swiglu") == std::string::npos && strcmp(kind, "ffn_swiglu") == 0) {
            // preferred node arrived second - drop what we already banked.
            d->tokE[il].clear();
            d->ntokens[il] = 0;
            d->layer_taken[il] = 0;
        } else {
            return true;
        }
    }

    if (d->swiglu_name_seen.empty()) {
        d->swiglu_name_seen = t->name;
    }

    if (t->type != GGML_TYPE_F32) {
        return true;
    }

    const int64_t n_ff     = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (n_ff <= 0 || n_tokens <= 0) {
        return true;
    }

    d->n_ff = n_ff;

    const int E = d->E;
    // Contiguous blocks only when E divides n_ff exactly. Otherwise fall back to
    // strided buckets, which still tile the pathways - better than dropping the
    // layer entirely and reporting a short table as if it were the whole model.
    const bool even = (n_ff % E == 0);
    if (!even && d->swiglu_name_seen == t->name) {
        fprintf(stderr, "[tomoe_probe] layer %d: n_ff=%lld not divisible by E=%d; "
                        "using strided buckets\n", il, (long long) n_ff, E);
    }

    std::vector<float> buf((size_t) n_ff * n_tokens);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));

    std::vector<float> & dst = d->tokE[il];
    dst.reserve(dst.size() + (size_t) n_tokens * E);

    for (int64_t j = 0; j < n_tokens; ++j) {
        const float * row = buf.data() + j * n_ff;
        for (int e = 0; e < E; ++e) {
            double acc = 0.0;
            for (int64_t i = e; i < n_ff; i += E) {
                acc += (double) row[i] * (double) row[i];
            }
            dst.push_back((float) acc);
        }
    }
    d->ntokens[il] += n_tokens;
    d->layer_taken[il] = 1;

    return true;
}

static void report(const probe_data & d) {
    const int E = d.E;
    const int k = std::min(d.topk, E);

    int captured = 0;
    for (char c : d.layer_taken) {
        captured += c ? 1 : 0;
    }
    if (captured == 0) {
        // Previously this printed nothing at all and exited 0, which reads as a
        // result. A model whose FFN nodes are all named something else must be
        // a loud failure, not an empty table.
        fprintf(stderr, "\n[tomoe_probe] ERROR: captured no FFN nodes.\n"
                        "  Looked for ffn_swiglu-<il> and ffn_out-<il>; saw %zu distinct node names, e.g.:\n",
                d.nodes_seen.size());
        for (size_t i = 0; i < d.nodes_seen.size() && i < 15; ++i) {
            fprintf(stderr, "    %s\n", d.nodes_seen[i].c_str());
        }
        fprintf(stderr, "  Add the architecture's node name to ff_node_kind() in this file.\n");
        return;
    }

    printf("\n=== ToMoE Phase-0 routing spread ===\n");
    printf("E=%d (bucketed), top-k=%d, n_ff=%lld, FFN node: %s, layers captured: %d\n",
           E, k, (long long) d.n_ff, d.swiglu_name_seen.c_str(), captured);
    printf("%5s %8s %9s %9s %9s %9s %9s %9s\n",
           "layer", "tokens", "top1", "oracleK", "globalK", "gain", "H/nats", "effE");

    double s_top1 = 0, s_oracle = 0, s_global = 0, s_ent = 0;
    int n_layers = 0;

    // sweep over k: mean share of per-token energy captured, across all layers
    std::vector<double> sw_oracle(E + 1, 0.0), sw_global(E + 1, 0.0);
    double sw_tokens = 0.0;

    for (int il = 0; il < (int) d.tokE.size(); ++il) {
        const int64_t nt = d.ntokens[il];
        if (nt <= 0) {
            continue;
        }
        const std::vector<float> & te = d.tokE[il];

        std::vector<double> agg(E, 0.0);
        for (int64_t j = 0; j < nt; ++j) {
            for (int e = 0; e < E; ++e) {
                agg[e] += te[(size_t) j * E + e];
            }
        }
        std::vector<int> order(E);
        for (int e = 0; e < E; ++e) {
            order[e] = e;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b2) { return agg[a] > agg[b2]; });

        double top1 = 0, oracle = 0, global = 0, ent = 0;
        for (int64_t j = 0; j < nt; ++j) {
            const float * row = te.data() + (size_t) j * E;
            double tot = 0.0;
            for (int e = 0; e < E; ++e) {
                tot += row[e];
            }
            if (tot <= 0.0) {
                continue;
            }
            std::vector<int> ord(E);
            for (int e = 0; e < E; ++e) {
                ord[e] = e;
            }
            std::sort(ord.begin(), ord.end(), [&](int a, int b2) { return row[a] > row[b2]; });

            top1 += row[ord[0]] / tot;
            for (int r = 0; r < k; ++r) {
                oracle += row[ord[r]] / tot;
            }
            for (int r = 0; r < k; ++r) {
                global += row[order[r]] / tot;
            }
            {
                double oc = 0.0, gc = 0.0;
                for (int r = 0; r < E; ++r) {
                    oc += row[ord[r]] / tot;
                    gc += row[order[r]] / tot;
                    sw_oracle[r + 1] += oc;
                    sw_global[r + 1] += gc;
                }
                sw_tokens += 1.0;
            }
            double h = 0.0;
            for (int e = 0; e < E; ++e) {
                const double p = row[e] / tot;
                if (p > 1e-12) {
                    h -= p * std::log(p);
                }
            }
            ent += h;
        }
        top1 /= nt; oracle /= nt; global /= nt; ent /= nt;

        printf("%5d %8lld %9.3f %9.3f %9.3f %9.3f %9.3f %9.1f\n",
               il, (long long) nt, top1, oracle, global, oracle - global, ent, std::exp(ent));

        s_top1 += top1; s_oracle += oracle; s_global += global; s_ent += ent;
        ++n_layers;
    }

    if (n_layers > 0) {
        printf("%5s %8s %9.3f %9.3f %9.3f %9.3f %9.3f %9.1f\n", "avg", "-",
               s_top1 / n_layers, s_oracle / n_layers, s_global / n_layers,
               (s_oracle - s_global) / n_layers, s_ent / n_layers, std::exp(s_ent / n_layers));
        printf("\ninterpretation: oracleK >> globalK means per-token routing helps.\n");
        printf("oracleK ~ globalK means the aggregate imatrix already captures what tokens use.\n");

        if (sw_tokens > 0.0) {
            printf("\n=== capture vs kept fraction (all layers) ===\n");
            printf("%5s %8s %12s %12s\n", "k", "kept%", "oracle", "global");
            for (int r = 1; r <= E; ++r) {
                printf("%5d %7.1f%% %12.3f %12.3f\n",
                       r, 100.0 * r / E, sw_oracle[r] / sw_tokens, sw_global[r] / sw_tokens);
            }
        }
    }
}

int main(int argc, char ** argv) {
    setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    probe_data d;
    if (const char * e = getenv("PROBE_E")) {
        int v = atoi(e);
        if (v >= 2) d.E = v;
    }
    if (const char * k = getenv("PROBE_TOPK")) {
        int v = atoi(k);
        if (v >= 1) d.topk = v;
    }

    params.cb_eval           = probe_eval;
    params.cb_eval_user_data = &d;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    // Under streaming Guanaco chains its router hook with this callback instead
    // of replacing it, so the probe sees the same nodes either way.

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        fprintf(stderr, "no input tokens; pass -p or -f\n");
        return 1;
    }

    int n_max = (int) tokens.size();
    if (const char * n = getenv("PROBE_TOKENS")) {
        int v = atoi(n);
        if (v > 0 && v < n_max) {
            n_max = v;
        }
    }

    fprintf(stderr, "[tomoe_probe] decoding %d tokens, E=%d topk=%d\n", n_max, d.E, d.topk);

    auto wall0 = std::chrono::steady_clock::now();

    int n_past = 0;
    int n_batch = (int) params.n_batch > 0 ? (int) params.n_batch : 512;
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

    auto wall1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(wall1 - wall0).count();

    long vmhwm_kb = 0;
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmHWM:", 6) == 0) {
                vmhwm_kb = atol(line + 6);
                break;
            }
        }
        fclose(f);
    }
    fprintf(stderr, "[tomoe_probe] decode %d tokens in %.2f s = %.3f tok/s, peak RSS %.0f MiB\n",
            n_past, secs, n_past / secs, vmhwm_kb / 1024.0);

    report(d);

    llama_backend_free();
    return 0;
}
