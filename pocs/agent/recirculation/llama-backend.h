#pragma once
#include "runner.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-adapter.h"
#include <memory>
#include <cstring>

namespace recirculation {
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
inline context_ptr make_context(llama_model * model, uint32_t capacity) {
    auto params = llama_context_default_params();
    params.n_ctx = capacity; params.n_seq_max = 1; params.n_batch = 1; params.n_ubatch = 1;
    params.offload_kqv = false; params.op_offload = false;
    params.n_threads = 1; params.n_threads_batch = 1;
    params.no_perf = false;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}
// This adapter exclusively owns a fresh CPU context. No scheduler/context shifts,
// external hooks, multiple sequences, or prompt-state reuse are admitted.
class llama_backend final : public backend {
public:
    llama_backend(llama_model * model, const recirculation_config & config, uint32_t capacity)
        : ctx(make_context(model, capacity)), config(config), width(llama_model_n_embd(model)),
          vocab_size(llama_vocab_n_tokens(llama_model_get_vocab(model))) {
        char arch[64] = {};
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 ||
                !recirculation_supported_architecture(arch) || !ctx || !llama_get_causal_attn(ctx.get()) ||
                !llama_memory_supports_rewind_last(llama_get_memory(ctx.get()))) {
            throw std::runtime_error("requires Qwen2, Qwen3 or Phi3 and exclusive causal KV replay support");
        }
        if (!recirculation_config_valid(config, llama_model_n_layer(model), width)) throw std::invalid_argument("invalid config");
    }
    bool ready(int32_t position) override {
        return ctx && position >= 0 && uint32_t(position) < llama_n_ctx(ctx.get()) &&
            llama_memory_supports_rewind_last(llama_get_memory(ctx.get())) &&
            llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == position - 1;
    }
    bool decode(int32_t token, int32_t position) {
        if (!ctx || token < 0 || token >= vocab_size) return false;
        llama_pos pos = position;
        llama_seq_id sequence = 0;
        llama_seq_id * sequences = &sequence;
        int32_t count = 1;
        int8_t output = 1;
        llama_batch batch{};
        batch.n_tokens = 1; batch.token = &token; batch.pos = &pos;
        batch.n_seq_id = &count; batch.seq_id = &sequences; batch.logits = &output;
        const int status = llama_decode(ctx.get(), batch);
        llama_synchronize(ctx.get());
        return status == 0;
    }
    bool first(int32_t token, int32_t position, bool capture_inputs, capture & result) override {
        if (capture_inputs && !capturing) {
            capturing = true;
            llama_set_embeddings_layer_inp(ctx.get(), config.source_layer_input, true);
            llama_set_embeddings_layer_inp(ctx.get(), config.destination_layer_input, true);
        }
        if (!decode(token, position)) return false;
        auto * logits = llama_get_logits_ith(ctx.get(), 0);
        if (!logits) return false;
        result.logits.assign(logits, logits + vocab_size);
        if (capture_inputs) {
            auto * d = llama_get_embeddings_layer_inp_row(ctx.get(), config.destination_layer_input, 0, position);
            auto * s = llama_get_embeddings_layer_inp_row(ctx.get(), config.source_layer_input, 0, position);
            if (!d || !s) return false;
            result.destination.assign(d, d + width); result.source.assign(s, s + width);
        }
        return true;
    }
    bool rewind(int32_t position) override {
        llama_synchronize(ctx.get());
        return llama_memory_rewind_last(llama_get_memory(ctx.get()), 0, position);
    }
    bool second(int32_t token, int32_t position, const std::vector<float> & patch, bool & matched) override {
        return patched(token, position, patch, matched, nullptr);
    }
    bool patched(int32_t token, int32_t position, const std::vector<float> & patch, bool & matched, capture * result) {
        last_patch = patch;
        ref = {}; matched = false;
        if (result) {
            capturing = true;
            llama_set_embeddings_layer_inp(ctx.get(), config.source_layer_input, true);
        }
        ref.operation = llama_residual_patch_operation::replace;
        ref.layer = config.destination_layer_input; ref.absolute_position = position;
        ref.sequence_id = 0; ref.values = patch.data(); ref.n_embd = patch.size(); ref.applied = &matched;
        llama_set_residual_patch(ctx.get(), &ref);
        // Keep the patch pointer and matched flag alive through synchronization.
        const bool success = decode(token, position);
        llama_set_residual_patch(ctx.get(), nullptr);
        if (success && result) {
            auto * source = llama_get_embeddings_layer_inp_row(ctx.get(), config.source_layer_input, 0, position);
            auto * logits = llama_get_logits_ith(ctx.get(), 0);
            if (!source || !logits) return false;
            result->source.assign(source, source + width);
            result->logits.assign(logits, logits + vocab_size);
        }
        return success;
    }
    void clear_hooks() noexcept override {
        if (!ctx) return;
        llama_set_residual_patch(ctx.get(), nullptr);
        if (capturing) {
            llama_set_embeddings_layer_inp(ctx.get(), config.source_layer_input, false);
            llama_set_embeddings_layer_inp(ctx.get(), config.destination_layer_input, false);
            capturing = false;
        }
    }
    void discard() noexcept override { ctx.reset(); }
    std::vector<float> last_patch;
    llama_context * context() const { return ctx.get(); }
private:
    context_ptr ctx;
    const recirculation_config config;
    const size_t width;
    const int32_t vocab_size;
    llama_residual_patch_ref ref;
    bool capturing = false;
};
}
