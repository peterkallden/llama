#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// PoC-only: capture the activation immediately before a layer's FFN down
// projection.  The returned rows are token-major and have n_ff(lid) columns.
LLAMA_API void llama_set_embeddings_ffn_down_inp(struct llama_context * ctx, uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_ffn_down_inp(struct llama_context * ctx, uint32_t lid);
// PoC-only: capture the output of a layer's FFN down projection, before the
// residual path. Rows are token-major and have n_embd columns.
LLAMA_API void llama_set_embeddings_ffn_down_out(struct llama_context * ctx, uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_ffn_down_out(struct llama_context * ctx, uint32_t lid);
// PoC-only CPU replay hook. Replaces one layer's FFN-down output with a
// token-major F32 buffer for a matching batch. This is not a scheduler or
// Vulkan integration path and is intended only for model-level validation.
LLAMA_API bool llama_set_ffn_down_output_override(struct llama_context * ctx, uint32_t lid,
                                                  const float * data, uint32_t n_tokens, uint32_t columns);

// Experimental runtime bridge for prepared compressed FFN-down artifacts.
// `is_ready` is evaluated while a graph is built; false preserves the normal
// GGUF matmul. `run` receives and returns token-major F32 data for a layer
// accepted by `is_ready`. The representation and cache policy remain behind
// this boundary, so D1/D2 do not leak into llama's graph construction.
typedef bool (*llama_ffn_down_runtime_is_ready_fn)(
        void * user_data, uint32_t lid, uint32_t input_columns, uint32_t output_columns);
typedef bool (*llama_ffn_down_runtime_run_fn)(
        void * user_data, uint32_t lid,
        const float * input, uint32_t n_tokens, uint32_t input_columns,
        float * output, uint32_t output_columns);
// Optional graph-time native binding hook. `node` is the generated runtime
// node for layer `lid`; the callback may register it with a backend-native
// dispatcher. Returning false keeps the existing CPU custom-op path.
typedef bool (*llama_ffn_down_runtime_native_bind_fn)(
        void * user_data, struct ggml_tensor * node, uint32_t lid);
typedef void (*llama_ffn_down_runtime_native_generation_begin_fn)(void * user_data);
LLAMA_API bool llama_set_ffn_down_runtime_provider(
        struct llama_context * ctx,
        llama_ffn_down_runtime_is_ready_fn is_ready,
        llama_ffn_down_runtime_run_fn run,
        void * user_data);
LLAMA_API bool llama_set_ffn_down_runtime_native_binding(
        struct llama_context * ctx,
        llama_ffn_down_runtime_native_bind_fn native_bind);
LLAMA_API bool llama_set_ffn_down_runtime_native_generation_begin(
        struct llama_context * ctx,
        llama_ffn_down_runtime_native_generation_begin_fn generation_begin);

// Generic tensor capture/runtime bridge for rank-2 model tensors.  Unlike the
// legacy FFN-down API this is keyed by the authoritative GGUF tensor name, so
// discovery/replay can target attention, FFN-up/gate, output, or other matrix
// roles without adding another layer-indexed API.
LLAMA_API void llama_set_embeddings_tensor(struct llama_context * ctx, const char * tensor_name, bool value);
LLAMA_API float * llama_get_embeddings_tensor(struct llama_context * ctx, const char * tensor_name);
LLAMA_API uint32_t llama_get_embeddings_tensor_columns(const struct llama_context * ctx, const char * tensor_name);

typedef bool (*llama_tensor_runtime_is_ready_fn)(
        void * user_data, const char * tensor_name, uint32_t input_columns, uint32_t output_columns);
typedef bool (*llama_tensor_runtime_run_fn)(
        void * user_data, const char * tensor_name,
        const float * input, uint32_t n_tokens, uint32_t input_columns,
        float * output, uint32_t output_columns);
typedef bool (*llama_tensor_runtime_native_bind_fn)(
        void * user_data, struct ggml_tensor * node, const char * tensor_name);
typedef void (*llama_tensor_runtime_generation_begin_fn)(void * user_data);

LLAMA_API bool llama_set_tensor_runtime_provider(
        struct llama_context * ctx,
        llama_tensor_runtime_is_ready_fn is_ready,
        llama_tensor_runtime_run_fn run,
        void * user_data);
LLAMA_API bool llama_set_tensor_runtime_native_binding(
        struct llama_context * ctx,
        llama_tensor_runtime_native_bind_fn native_bind);
LLAMA_API bool llama_set_tensor_runtime_native_generation_begin(
        struct llama_context * ctx,
        llama_tensor_runtime_generation_begin_fn generation_begin);

// PoC helper exposing the FFN width needed to interpret the capture above.
LLAMA_API int32_t llama_model_n_ff(const struct llama_model * model, uint32_t layer);
// Returns the logical shape of a named rank-2 model tensor.  `columns` is
// ne[0] (the matmul input width) and `rows` is ne[1] (the output width).
LLAMA_API bool llama_model_get_tensor_shape(const struct llama_model * model,
                                            const char * tensor_name,
                                            uint32_t * columns,
                                            uint32_t * rows);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
