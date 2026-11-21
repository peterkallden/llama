#include "llama-kv-cache.h"
#include "models/models.h"
#include "llama-ext.h"
#include <cassert>
#include <limits>

int main() {
    // Filtered-layer fixture exercises production metadata operations without model weights.
    llama_model_qwen3 model(llama_model_default_params());
    model.hparams = {};
    model.hparams.n_layer_all = 1;
    auto make = [&](unsigned sequences, unsigned swa, llama_memory_t other = nullptr) {
        return std::make_unique<llama_kv_cache>(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, 16, sequences, 1, swa,
            swa ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_NONE, other,
            [](int32_t) { return false; }, llama_memory_i::layer_reuse_cb{}, llama_memory_i::layer_share_cb{});
    };
    auto cache = make(1, 0);
    assert(llama_memory_supports_rewind_last(cache.get()));
    assert(!llama_memory_can_rewind_last(cache.get(), 0, 0));
    auto & cells = const_cast<llama_kv_cells &>(cache->get_cells(0));
    cells.pos_set(0, 0); cells.seq_add(0, 0);
    cells.pos_set(1, 1); cells.seq_add(1, 0);
    assert(llama_memory_can_rewind_last(cache.get(), 0, 1));
    assert(!llama_memory_rewind_last(cache.get(), 1, 1));
    assert(!llama_memory_rewind_last(cache.get(), 0, 0));
    assert(llama_memory_rewind_last(cache.get(), 0, 1));
    assert(cache->seq_pos_max(0) == 0 && cells.is_empty(1));
    assert(!llama_memory_rewind_last(cache.get(), 0, 1));
    cells.pos_set(1, 1); cells.seq_add(1, 0);
    assert(llama_memory_rewind_last(cache.get(), 0, 1));
    assert(!llama_memory_can_rewind_last(cache.get(), 0, std::numeric_limits<llama_pos>::max()));
    cells.seq_add(0, 1);
    assert(!llama_memory_supports_rewind_last(cache.get()));
    cells.seq_rm(0, 1);
    {
        auto shared = make(1, 0, cache.get());
        assert(!llama_memory_supports_rewind_last(cache.get()));
        assert(!llama_memory_supports_rewind_last(shared.get()));
    }
    assert(llama_memory_supports_rewind_last(cache.get()));
    auto multiple = make(2, 0);
    auto swa = make(1, 8);
    assert(!llama_memory_supports_rewind_last(multiple.get()));
    assert(!llama_memory_supports_rewind_last(swa.get()));
    assert(!llama_memory_supports_rewind_last(nullptr));
    cache->clear(false);
    cells.pos_set(0, 0); cells.seq_add(0, 0);
    cache->seq_add(0, 0, -1, 1);
    assert(!llama_memory_supports_rewind_last(cache.get()));
}
