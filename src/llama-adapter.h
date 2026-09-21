#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

struct llama_ubatch;

// Non-owning graph hook for a backend that can apply different control
// vectors to different sequences in one ubatch. The callback runs while the
// graph is being built and must return `cur` unchanged when no overlay is
// selected for the layer. It may construct backend tensors in `ctx`, but must
// not execute the graph or retain `ctx`/`cur` after the callback returns.
// The scalar cvec path remains the default; the backend owns the table,
// device buffers and lifetime.
using llama_adapter_cvec_batch_apply_fn = ggml_tensor * (*) (
        ggml_context * ctx,
        ggml_tensor * cur,
        int il,
        const llama_ubatch & ubatch,
        void * user_data);

struct llama_adapter_cvec_batch_ref {
    // The reference and user_data must remain valid through graph execution.
    llama_adapter_cvec_batch_apply_fn apply = nullptr;
    void * user_data = nullptr;
};

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

// Backend-resident per-sequence control-vector table. Row order follows
// seq_ids; the graph callback selects the row for each token in the current
// ubatch. This is the low-level implementation behind
// llama_adapter_cvec_batch_ref. It is deliberately unaware of FlyDelta,
// server slots and search policy.
struct llama_adapter_cvec_batch {
    llama_adapter_cvec_batch() = default;
    ~llama_adapter_cvec_batch() = default;

    llama_adapter_cvec_batch(const llama_adapter_cvec_batch &) = delete;
    llama_adapter_cvec_batch & operator=(const llama_adapter_cvec_batch &) = delete;
    llama_adapter_cvec_batch(llama_adapter_cvec_batch &&) = delete;
    llama_adapter_cvec_batch & operator=(llama_adapter_cvec_batch &&) = delete;

    bool apply(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const std::vector<const float *> & data,
            size_t data_len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

    // Sparse counterpart for device-aware backends. Each row contains only
    // the layers listed in layer_indices, in sorted layer-major order.
    bool apply_sparse(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const std::vector<const float *> & data,
            size_t data_len,
            int32_t n_embd,
            const std::vector<uint32_t> & layer_indices);

    void clear();

    bool enabled() const { return active; }

    const llama_adapter_cvec_batch_ref & ref() const { return batch_ref; }

private:
    static ggml_tensor * apply_callback(
            ggml_context * ctx,
            ggml_tensor * cur,
            int il,
            const llama_ubatch & ubatch,
            void * user_data);

    bool init(
            const llama_model & model,
            int32_t n_embd,
            size_t n_seq,
            const std::vector<uint32_t> & layer_indices);
    bool apply_impl(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const std::vector<const float *> & data,
            size_t data_len,
            int32_t n_embd,
            const std::vector<uint32_t> & layer_indices,
            bool dense_layout);
    ggml_tensor * tensor_for(int il) const;
    int32_t row_for_token(const llama_ubatch & ubatch, uint32_t token_index) const;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;
    int32_t n_embd      = 0;
    bool active         = false;

    std::vector<llama_seq_id> seq_ids;
    // The row-selector is a small host-resident graph input.  It cannot be
    // written through ggml_set_i32_1d() on the graph metadata context because
    // that context is created with no_alloc=true.  A selector is kept per
    // graph context: concurrent server slots must never rewrite one another's
    // indices while their graphs are being built or executed.
    struct selector_state {
        ggml_context_ptr ctx;
        ggml_tensor * tensor = nullptr;
        ggml_backend_buffer_ptr buf;
        uint32_t count = 0;
    };
    mutable std::mutex selector_mutex;
    std::unordered_map<ggml_context *, selector_state> selector_states;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<ggml_tensor *> tensors;

    llama_adapter_cvec_batch_ref batch_ref {
        /* .apply = */ apply_callback,
        /* .user_data = */ this,
    };
};

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
