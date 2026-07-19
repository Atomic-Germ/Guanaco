#pragma once

#include "guanaco/guanaco.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wrap an existing CPU ggml backend so that, after every whole-graph compute,
// the MoE router (ffn_moe_topk-<il>) outputs are observed via `hook`. Unlike
// ggml_backend_sched_set_eval_callback, this does NOT force the scheduler into
// per-node execution (which breaks mul_mat_id), so inference stays correct.
// The returned backend owns and frees `cpu`; free it with ggml_backend_free.
GUANACO_API ggml_backend_t guanaco_wrap_cpu_backend(ggml_backend_t cpu, void* hook);

#ifdef __cplusplus
}
#endif
