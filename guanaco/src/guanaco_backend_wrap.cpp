// GuanacoRouterBackend: a thin wrapper around the CPU backend that lets us
// observe the MoE router (ffn_moe_topk-<il>) outputs without forcing the
// scheduler into its slow per-node eval-callback mode.
//
// Why not ggml_backend_sched_set_eval_callback? Registering any eval callback
// makes the scheduler compute the graph one node (or tiny slice) at a time
// with a synchronize between each. For MoE models that breaks mul_mat_id
// (its workspace/row grouping assumes a whole-graph view), producing garbage
// output. By patching the backend's graph_compute in place instead, the graph
// is still computed as a single whole-graph pass (correct), and we simply scan
// the node list after the real backend is done to read the router ids.

#include "guanaco/guanaco.h"
#include "guanaco/guanaco_model_hook.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include <cstring>
#include <string>

// We patch the real CPU backend's vtable in place, so we keep the original
// graph_compute and the hook in globals (there is one model load at a time).
static guanaco::GuanacoModelHook * g_hook = nullptr;
static enum ggml_status (*g_orig_graph_compute)(ggml_backend_t, struct ggml_cgraph *) = nullptr;

// Scan the just-computed graph for router (ffn_moe_topk-<il>) nodes and hand
// the selected expert ids to the Guanaco hook. This runs AFTER the whole graph
// has been computed on the real backend, so mul_mat_id already used the
// (resident) expert weights - correctness is unaffected. The ids drive our
// madvise-based cache warming / LRU eviction / pilot prefetch.
static void guanaco_scan_router_nodes(const struct ggml_cgraph * g) {
    if (g_hook == nullptr || g == nullptr) return;
    const int n_nodes = ggml_graph_n_nodes(const_cast<struct ggml_cgraph *>(g));
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * t = ggml_graph_node(const_cast<struct ggml_cgraph *>(g), i);
        if (t == nullptr) continue;
        // During the scheduler's measure/alloc graph passes the tensors are
        // not yet allocated (t->data == nullptr). Skip those - we only want
        // the router output of a real, allocated graph compute.
        if (t->data == nullptr) continue;
        const char * name = ggml_get_name(t);
        if (name == nullptr || strncmp(name, "ffn_moe_topk", 12) != 0) continue;
        int layer = -1;
        const char * dash = strrchr(name, '-');
        if (dash != nullptr) layer = atoi(dash + 1);
        const int n = (int)(t->ne[0] * t->ne[1]); // n_expert_used * n_tokens
        g_hook->on_router_computed(layer, static_cast<const int *>(t->data), n);
    }
}

static enum ggml_status guanaco_wrapped_graph_compute(ggml_backend_t b, struct ggml_cgraph * g) {
    enum ggml_status rc = g_orig_graph_compute(b, g);
    guanaco_scan_router_nodes(g);
    return rc;
}

// Patch the real CPU backend so its graph_compute first runs the original
// compute, then scans the (now filled) router output tensors. Returns the same
// `cpu` backend pointer (mutated in place) so the caller keeps using it; the
// scheduler's backend pointer stays valid.
extern "C" GUANACO_API ggml_backend_t guanaco_wrap_cpu_backend(ggml_backend_t cpu, void * hook) {
    if (cpu == nullptr) return nullptr;
    g_hook = static_cast<guanaco::GuanacoModelHook *>(hook);
    g_orig_graph_compute = cpu->iface.graph_compute;
    cpu->iface.graph_compute = guanaco_wrapped_graph_compute;
    return cpu;
}
