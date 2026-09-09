#pragma once

#include "llama-ext.h"

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_set>

struct llama_ffn_down_output_override {
    const float * data = nullptr;
    uint32_t n_tokens = 0;
    uint32_t columns = 0;
};

struct llama_ffn_down_runtime_provider {
    llama_ffn_down_runtime_is_ready_fn is_ready = nullptr;
    llama_ffn_down_runtime_run_fn run = nullptr;
    // Optional graph-time hook. When it returns true, the provider has bound
    // the generated runtime node to a backend-native dispatch path. A false
    // result preserves the ordinary custom-op/CPU fallback.
    llama_ffn_down_runtime_native_bind_fn native_bind = nullptr;
    // Called once before a graph generation starts so external owners can
    // discard bindings for the previous graph generation.
    llama_ffn_down_runtime_native_generation_begin_fn native_generation_begin = nullptr;
    void * user_data = nullptr;
};

// Generic tensor runtime bridge used by the ASTC research path.  The
// legacy FFN-down provider remains source-compatible; this bridge adds the
// authoritative tensor name so cache/replay can target any rank-2 matrix.
struct llama_tensor_runtime_provider {
    llama_tensor_runtime_is_ready_fn is_ready = nullptr;
    llama_tensor_runtime_run_fn run = nullptr;
    llama_tensor_runtime_native_bind_fn native_bind = nullptr;
    llama_tensor_runtime_generation_begin_fn generation_begin = nullptr;
    void * user_data = nullptr;
};

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    // PoC-only capture of the activation immediately before each layer's FFN
    // down projection.  This is deliberately separate from layer inputs: for
    // gated FFNs its width is n_ff(il), which may be larger than n_embd.
    std::vector<bool> embeddings_ffn_down_inp;
    std::vector<bool> embeddings_ffn_down_out;
    std::unordered_set<std::string> embeddings_tensor_names;
    llama_tensor_runtime_provider tensor_runtime_provider;
    std::vector<llama_ffn_down_output_override> ffn_down_output_overrides;
    llama_ffn_down_runtime_provider ffn_down_runtime_provider;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
