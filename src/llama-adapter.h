#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

struct llama_ubatch;

enum class llama_residual_patch_operation : uint8_t { replace, add };

// Non-owning, single-site graph intervention. It is deliberately unrelated
// to adapters/cvecs: callers provide a fully prepared vector.
struct llama_residual_patch_ref {
    llama_residual_patch_operation operation = llama_residual_patch_operation::add;
    uint32_t layer = 0;
    int32_t absolute_position = -1;
    llama_seq_id sequence_id = -1;
    const float * values = nullptr;
    size_t n_embd = 0;
    bool * applied = nullptr;
};

// Request-scoped, backend-neutral compact diagnostics. The graph resolves
// the two sequence/position rows and computes the scalar reductions while
// the layer inputs remain device-resident. The caller owns the referenced
// direction data and receives the scalar results in-place.
struct llama_layer_reduction_entry {
    uint32_t layer = 0;
    llama_seq_id baseline_sequence_id = -1;
    llama_seq_id overlay_sequence_id = -1;
    int32_t absolute_position = -1;
    const float * direction = nullptr;
    size_t n_embd = 0;

    float dot_shift_delta = 0.0f;
    float shift_squared = 0.0f;
    float delta_squared = 0.0f;
    float residual_squared = 0.0f;
    bool available = false;
    std::string failure_reason;
};

struct llama_layer_reduction_ref {
    std::vector<llama_layer_reduction_entry> entries;
};

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

    LLAMA_API bool apply(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const std::vector<const float *> & data,
            size_t data_len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

    // Sparse counterpart for device-aware backends. Each row contains only
    // the layers listed in layer_indices, in sorted layer-major order.
    LLAMA_API bool apply_sparse(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const std::vector<const float *> & data,
            size_t data_len,
            int32_t n_embd,
            const std::vector<uint32_t> & layer_indices);

    // Native factorized counterpart. `basis` is layer-major [layer][rank][d]
    // and is shared by all sequence rows; each coefficient row has `rank`
    // FP32 values.  No full d-dimensional row is materialized on the host.
    LLAMA_API bool apply_factorized_sparse(
            const llama_model & model,
            const std::vector<llama_seq_id> & seq_ids,
            const float * basis,
            size_t basis_len,
            const std::vector<const float *> & coefficients,
            int32_t rank,
            int32_t n_embd,
            const std::vector<uint32_t> & layer_indices);

    // Compose one backend-resident dense overlay row per sequence before the
    // model graph runs. The basis is cached by fingerprint/layout; only the
    // coefficient matrix and composed rows are wave-scoped.
    LLAMA_API bool apply_factorized_composed(
            const llama_model & model,
            ggml_backend_sched_t scheduler,
            const std::vector<llama_seq_id> & seq_ids,
            const std::string & basis_fingerprint,
            const float * basis,
            size_t basis_len,
            const std::vector<const float *> & coefficients,
            int32_t rank,
            int32_t n_embd,
            const std::vector<uint32_t> & layer_indices);

    LLAMA_API void clear();
    // Release only the current wave binding while preserving the one-entry
    // resident basis cache for the next compatible coefficient wave.
    LLAMA_API void clear_binding();

    bool enabled() const { return active; }

    // Telemetry: true when at least one per-layer table buffer is owned by a
    // non-CPU backend. The graph callback remains backend-neutral.
    bool device_resident() const { return device_resident_flag; }
    bool resident_basis_cache_hit() const { return last_basis_cache_hit; }
    bool composed_overlay_cache_hit() const { return last_composed_overlay_cache_hit; }
    size_t resident_basis_upload_count() const { return basis_upload_count; }
    size_t last_basis_upload_count() const { return last_basis_upload_count_value; }
    uint64_t last_composition_us() const { return last_composition_us_value; }
    size_t resident_basis_bytes() const { return resident_basis_bytes_value; }
    size_t composed_overlay_bytes() const { return composed_overlay_bytes_value; }
    bool callback_applied() const { return callback_applied_flag.load(); }
    std::string callback_failure_reason() const;
    // Start a fresh graph observation without discarding device buffers. The
    // server calls this before each decode so a callback from an earlier graph
    // cannot make a later no-op graph look successful.
    void begin_graph_observation();

    const llama_adapter_cvec_batch_ref & ref() const { return batch_ref; }

private:
    static LLAMA_API ggml_tensor * apply_callback(
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
    int32_t row_for_sequence(llama_seq_id sequence_id) const;
    int32_t row_for_token(const llama_ubatch & ubatch, uint32_t token_index) const;
    void mark_callback_failure(const std::string & reason);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;
    int32_t n_embd      = 0;
    bool active         = false;
    bool device_resident_flag = false;
    std::atomic<bool> callback_applied_flag{false};
    std::atomic<bool> callback_failed_flag{false};
    mutable std::mutex callback_error_mutex;
    std::string callback_error;

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
    struct resident_basis_layer {
        ggml_backend_ptr backend;
        ggml_context_ptr ctx;
        ggml_tensor * tensor = nullptr;
        ggml_backend_buffer_ptr buffer;
        ggml_backend_buffer_type_t buft = nullptr;
    };
    std::string resident_basis_key;
    std::vector<resident_basis_layer> resident_basis;
    std::string composed_overlay_key;
    std::string composed_basis_key;
    std::vector<ggml_context_ptr> composed_ctxs;
    std::vector<ggml_backend_buffer_ptr> composed_bufs;
    std::vector<ggml_tensor *> composed_tensors;
    std::unordered_map<std::string, int32_t> composed_row_by_coefficients;
    int32_t composed_zero_overlay_row = -1;
    std::vector<int32_t> seq_overlay_rows;
    int32_t zero_overlay_row = -1;
    size_t resident_basis_bytes_value = 0;
    bool last_basis_cache_hit = false;
    bool last_composed_overlay_cache_hit = false;
    size_t basis_upload_count = 0;
    size_t last_basis_upload_count_value = 0;
    uint64_t last_composition_us_value = 0;
    size_t composed_overlay_bytes_value = 0;
    std::vector<ggml_tensor *> tensors;
    std::vector<ggml_tensor *> factorized_basis_tensors;
    std::vector<ggml_tensor *> factorized_coefficient_tensors;
    bool factorized = false;
    int32_t factorized_rank = 0;

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
