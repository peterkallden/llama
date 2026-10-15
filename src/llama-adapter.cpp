#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-batch.h"

#include <map>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cinttypes>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer());
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer(); il++) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer(); il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

ggml_tensor * llama_adapter_cvec_batch::tensor_for(int il) const {
    if (!active || il < 0 || il < layer_start || il > layer_end ||
            (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

bool llama_adapter_cvec_batch::init(
        const llama_model & model,
        const int32_t n_embd,
        const size_t n_seq,
        const std::vector<uint32_t> & layer_indices) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);
            return ctx;
        }

        return it->second;
    };

    tensors.assign(hparams.n_layer(), nullptr);
    for (const uint32_t layer : layer_indices) {
        if (layer < 1 || layer >= hparams.n_layer()) {
            LLAMA_LOG_ERROR("%s: invalid layer in batch control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_type_t buft = model.select_buft(layer);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for batch control vector\n", __func__);
            return false;
        }

        ggml_tensor * tensor = ggml_new_tensor_2d(
            // Reserve one backend-zeroed fallback row. A resident server may
            // build a graph that contains an otherwise inactive sequence;
            // that sequence must receive no overlay, not another sequence's
            // sparse row and not a callback failure.
            ctx, GGML_TYPE_F32, n_embd, static_cast<int64_t>(n_seq + 1));
        tensors[layer] = tensor;
    }

    bufs.reserve(ctx_map.size());
    device_resident_flag = false;
    for (const auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate batch control-vector buffer\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        const ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        device_resident_flag = device_resident_flag ||
            (device != nullptr && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU);
        bufs.emplace_back(buf);
    }

    return true;
}

void llama_adapter_cvec_batch::clear() {
    clear_binding();
    composed_bufs.clear();
    composed_ctxs.clear();
    composed_tensors.clear();
    composed_row_by_coefficients.clear();
    composed_overlay_key.clear();
    composed_basis_key.clear();
    composed_zero_overlay_row = -1;
    seq_overlay_rows.clear();
    zero_overlay_row = -1;
    resident_basis.clear();
    resident_basis_key.clear();
    resident_basis_bytes_value = 0;
    basis_upload_count = 0;
    last_basis_upload_count_value = 0;
    last_basis_cache_hit = false;
    last_composition_us_value = 0;
    composed_overlay_bytes_value = 0;
    last_composed_overlay_cache_hit = false;
}

void llama_adapter_cvec_batch::clear_binding() {
    active = false;
    device_resident_flag = false;
    layer_start = -1;
    layer_end = -1;
    n_embd = 0;
    seq_ids.clear();
    seq_overlay_rows.clear();
    zero_overlay_row = -1;
    {
        std::lock_guard<std::mutex> lock(selector_mutex);
        selector_states.clear();
    }
    tensors.clear();
    factorized_basis_tensors.clear();
    factorized_coefficient_tensors.clear();
    factorized = false;
    factorized_rank = 0;
    callback_applied_flag.store(false);
    callback_failed_flag.store(false);
    {
        std::lock_guard<std::mutex> lock(callback_error_mutex);
        callback_error.clear();
    }
    bufs.clear();
    ctxs.clear();
}

void llama_adapter_cvec_batch::mark_callback_failure(const std::string & reason) {
    callback_failed_flag.store(true);
    std::lock_guard<std::mutex> lock(callback_error_mutex);
    if (callback_error.empty()) callback_error = reason;
}

void llama_adapter_cvec_batch::begin_graph_observation() {
    callback_applied_flag.store(false);
    callback_failed_flag.store(false);
    std::lock_guard<std::mutex> lock(callback_error_mutex);
    callback_error.clear();
}

std::string llama_adapter_cvec_batch::callback_failure_reason() const {
    if (!callback_failed_flag.load()) return {};
    std::lock_guard<std::mutex> lock(callback_error_mutex);
    return callback_error.empty() ? "unspecified graph callback failure" : callback_error;
}

bool llama_adapter_cvec_batch::apply_factorized_sparse(
        const llama_model & model,
        const std::vector<llama_seq_id> & seq_ids,
        const float * basis,
        const size_t basis_len,
        const std::vector<const float *> & coefficients,
        const int32_t rank,
        const int32_t n_embd,
        const std::vector<uint32_t> & layer_indices) {
    clear();
    if (basis == nullptr || rank <= 0 || seq_ids.empty() || seq_ids.size() != coefficients.size() ||
            n_embd != static_cast<int32_t>(model.hparams.n_embd) || layer_indices.empty() ||
            basis_len != layer_indices.size() * static_cast<size_t>(rank) * static_cast<size_t>(n_embd) ||
            std::any_of(coefficients.begin(), coefficients.end(), [](const float * value) { return value == nullptr; })) return false;
    std::vector<llama_seq_id> sorted = seq_ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return false;
    if (std::adjacent_find(layer_indices.begin(), layer_indices.end(),
            [](uint32_t a, uint32_t b) { return a >= b; }) != layer_indices.end() ||
            layer_indices.front() < 1 || layer_indices.back() >= model.hparams.n_layer()) return false;

    const auto & hparams = model.hparams;
    std::map<ggml_backend_buffer_type_t, ggml_context *> contexts;
    auto context_for = [&](ggml_backend_buffer_type_t buft) {
        auto it = contexts.find(buft);
        if (it != contexts.end()) return it->second;
        ggml_init_params params = { hparams.n_layer()*3*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return static_cast<ggml_context *>(nullptr);
        contexts[buft] = ctx; ctxs.emplace_back(ctx); return ctx;
    };
    tensors.assign(hparams.n_layer(), nullptr);
    factorized_basis_tensors.assign(hparams.n_layer(), nullptr);
    factorized_coefficient_tensors.assign(hparams.n_layer(), nullptr);
    for (size_t i = 0; i < layer_indices.size(); ++i) {
        const uint32_t layer = layer_indices[i];
        ggml_context * ctx = context_for(model.select_buft(layer));
        if (!ctx) { clear(); return false; }
        // ggml_mul_mat requires the shared reduction dimension at ne[0] on
        // both inputs. The canonical host basis is rank-major [rank][n_embd],
        // while the graph operand must be [rank, n_embd] with rank contiguous.
        factorized_basis_tensors[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);
        factorized_coefficient_tensors[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, seq_ids.size() + 1);
        if (!factorized_basis_tensors[layer] || !factorized_coefficient_tensors[layer]) {
            clear();
            return false;
        }
    }
    device_resident_flag = false;
    for (const auto & [buft, ctx] : contexts) {
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buffer) { clear(); return false; }
        ggml_backend_buffer_clear(buffer, 0);
        const auto dev = ggml_backend_buft_get_device(buft);
        device_resident_flag = device_resident_flag || (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU);
        bufs.emplace_back(buffer);
    }
    for (size_t i = 0; i < layer_indices.size(); ++i) {
        const uint32_t layer = layer_indices[i];
        ggml_tensor * b = factorized_basis_tensors[layer];
        ggml_tensor * c = factorized_coefficient_tensors[layer];
        std::vector<float> graph_basis(static_cast<size_t>(n_embd) * rank);
        const float * source_basis = basis + i * static_cast<size_t>(rank) * n_embd;
        for (int32_t hidden = 0; hidden < n_embd; ++hidden) {
            for (int32_t component = 0; component < rank; ++component) {
                graph_basis[static_cast<size_t>(hidden) * rank + component] =
                    source_basis[static_cast<size_t>(component) * n_embd + hidden];
            }
        }
        ggml_backend_tensor_set(b, graph_basis.data(), 0,
            static_cast<size_t>(rank) * n_embd * sizeof(float));
        for (size_t row = 0; row < coefficients.size(); ++row) {
            ggml_backend_tensor_set(c, coefficients[row], row * static_cast<size_t>(rank) * sizeof(float),
                static_cast<size_t>(rank) * sizeof(float));
        }
    }
    this->seq_ids = seq_ids; this->n_embd = n_embd;
    this->layer_start = static_cast<int32_t>(layer_indices.front());
    this->layer_end = static_cast<int32_t>(layer_indices.back());
    this->factorized = true; this->factorized_rank = rank; this->active = true;
    return true;
}

bool llama_adapter_cvec_batch::apply_factorized_composed(
        const llama_model & model,
        ggml_backend_sched_t scheduler,
        const std::vector<llama_seq_id> & seq_ids,
        const std::string & basis_fingerprint,
        const float * basis,
        const size_t basis_len,
        const std::vector<const float *> & coefficients,
        const int32_t rank,
        const int32_t n_embd,
        const std::vector<uint32_t> & layer_indices) {
    const auto compose_started = std::chrono::steady_clock::now();
    clear_binding();
    last_basis_cache_hit = false;
    last_composed_overlay_cache_hit = false;
    last_basis_upload_count_value = 0;
    last_composition_us_value = 0;
    if (scheduler == nullptr || basis == nullptr || basis_fingerprint.empty() || rank <= 0 || seq_ids.empty() ||
            seq_ids.size() != coefficients.size() ||
            n_embd != static_cast<int32_t>(model.hparams.n_embd) || layer_indices.empty() ||
            static_cast<size_t>(rank) > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(n_embd) ||
            layer_indices.size() > std::numeric_limits<size_t>::max() /
                (static_cast<size_t>(rank) * static_cast<size_t>(n_embd)) ||
            basis_len != layer_indices.size() * static_cast<size_t>(rank) * static_cast<size_t>(n_embd) ||
            std::any_of(coefficients.begin(), coefficients.end(), [](const float * value) { return value == nullptr; })) {
        return false;
    }
    std::vector<llama_seq_id> sorted = seq_ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
            std::adjacent_find(layer_indices.begin(), layer_indices.end(),
                [](uint32_t a, uint32_t b) { return a >= b; }) != layer_indices.end() ||
            layer_indices.front() < 1 || layer_indices.back() >= model.hparams.n_layer()) {
        return false;
    }

    for (const float * row : coefficients) {
        for (int32_t component = 0; component < rank; ++component) {
            if (!std::isfinite(row[component])) return false;
        }
    }

    std::ostringstream key_builder;
    key_builder << basis_fingerprint << ":rank=" << rank << ":d=" << n_embd << ":f32";
    for (const uint32_t layer : layer_indices) {
        const auto buft = model.select_buft(layer);
        const auto device = ggml_backend_buft_get_device(buft);
        key_builder << ":l=" << layer << ":buft=" << static_cast<const void *>(buft)
                    << ":dev=" << static_cast<const void *>(device);
    }
    const std::string requested_key = key_builder.str();
    bool cache_hit = resident_basis_key == requested_key && resident_basis.size() == layer_indices.size();
    last_basis_cache_hit = cache_hit;
    if (!cache_hit) {
        // The server synchronizes the model context before rebinding. Also
        // drain any direct backend streams before releasing the previous key.
        for (auto & cached : resident_basis) {
            if (cached.backend) ggml_backend_synchronize(cached.backend.get());
        }
        resident_basis.clear();
        resident_basis_key.clear();
        resident_basis_bytes_value = 0;
        resident_basis.reserve(layer_indices.size());
        const size_t one_basis_bytes = static_cast<size_t>(rank) * n_embd * sizeof(float);
        for (size_t i = 0; i < layer_indices.size(); ++i) {
            const uint32_t layer = layer_indices[i];
            const auto buft = model.select_buft(layer);
            const auto device = ggml_backend_buft_get_device(buft);
            if (device == nullptr) {
                resident_basis.clear();
                return false;
            }
            resident_basis_layer cached;
            cached.buft = buft;
            cached.backend.reset(ggml_backend_dev_init(device, nullptr));
            if (!cached.backend) {
                resident_basis.clear();
                return false;
            }
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            cached.ctx.reset(ggml_init(params));
            if (!cached.ctx) {
                resident_basis.clear();
                return false;
            }
            cached.tensor = ggml_new_tensor_2d(cached.ctx.get(), GGML_TYPE_F32, rank, n_embd);
            if (!cached.tensor) {
                resident_basis.clear();
                return false;
            }
            cached.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(cached.ctx.get(), buft));
            if (!cached.buffer) {
                resident_basis.clear();
                return false;
            }
            std::vector<float> graph_basis(static_cast<size_t>(n_embd) * rank);
            const float * source_basis = basis + i * static_cast<size_t>(rank) * n_embd;
            for (int32_t hidden = 0; hidden < n_embd; ++hidden) {
                for (int32_t component = 0; component < rank; ++component) {
                    graph_basis[static_cast<size_t>(hidden) * rank + component] =
                        source_basis[static_cast<size_t>(component) * n_embd + hidden];
                }
            }
            ggml_backend_tensor_set(cached.tensor, graph_basis.data(), 0, one_basis_bytes);
            ggml_backend_synchronize(cached.backend.get());
            resident_basis_bytes_value += one_basis_bytes;
            ++basis_upload_count;
            ++last_basis_upload_count_value;
            resident_basis.emplace_back(std::move(cached));
        }
        resident_basis_key = requested_key;
    }

    // Coefficient rows are canonicalized independently of sequence ordering.
    // The composed table can therefore survive fragments that contain a
    // different subset/order of the same logical wave's sequence rows.
    std::map<std::string, const float *> unique_coefficients;
    for (const float * row : coefficients) {
        unique_coefficients.emplace(
            std::string(reinterpret_cast<const char *>(row),
                static_cast<size_t>(rank) * sizeof(float)), row);
    }
    std::ostringstream overlay_key_builder;
    overlay_key_builder << requested_key << ":unique_rows=" << unique_coefficients.size();
    for (const auto & row : unique_coefficients) {
        overlay_key_builder.write(row.first.data(), static_cast<std::streamsize>(row.first.size()));
    }
    const std::string requested_overlay_key = overlay_key_builder.str();
    const bool overlay_cache_hit = composed_basis_key == requested_key &&
        composed_tensors.size() == model.hparams.n_layer() && !composed_bufs.empty() &&
        std::all_of(unique_coefficients.begin(), unique_coefficients.end(), [&](const auto & row) {
            return composed_row_by_coefficients.find(row.first) !=
                composed_row_by_coefficients.end();
        });
    last_composed_overlay_cache_hit = overlay_cache_hit;

    if (!overlay_cache_hit) {
        // The previous table may still be referenced by an asynchronously
        // submitted graph. Drain it before replacing any backend allocation.
        ggml_backend_sched_synchronize(scheduler);
        composed_bufs.clear();
        composed_ctxs.clear();
        composed_tensors.clear();
        composed_row_by_coefficients.clear();
        composed_overlay_key.clear();
        composed_basis_key.clear();
        composed_zero_overlay_row = -1;
        composed_overlay_bytes_value = 0;

        const auto & hparams = model.hparams;
        std::vector<ggml_context_ptr> next_contexts;
        std::vector<ggml_backend_buffer_ptr> next_buffers;
        std::vector<ggml_tensor *> next_tensors(hparams.n_layer(), nullptr);
        size_t next_overlay_bytes = 0;
        std::unordered_map<std::string, int32_t> next_row_by_coefficients;
        next_row_by_coefficients.reserve(unique_coefficients.size());
        int32_t next_row = 0;
        for (const auto & row : unique_coefficients) {
            next_row_by_coefficients.emplace(row.first, next_row++);
        }
        for (size_t i = 0; i < layer_indices.size(); ++i) {
            const uint32_t layer = layer_indices[i];
            const auto buft = model.select_buft(layer);
            ggml_init_params params = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(params);
            if (ctx) next_contexts.emplace_back(ctx);
            if (!ctx || buft == nullptr || ggml_backend_buft_get_device(buft) == nullptr) return false;
            const size_t row_count = unique_coefficients.size() + 1;
            ggml_tensor * coeff_tensor = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, rank, static_cast<int64_t>(row_count));
            if (!coeff_tensor) return false;
            ggml_tensor * projected = ggml_mul_mat(ctx, resident_basis[i].tensor, coeff_tensor);
            if (!projected) return false;
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buffer) return false;
            ggml_backend_buffer_clear(buffer, 0);
            size_t row_index = 0;
            for (const auto & row : unique_coefficients) {
                ggml_backend_tensor_set(coeff_tensor, row.second,
                    row_index++ * static_cast<size_t>(rank) * sizeof(float),
                    static_cast<size_t>(rank) * sizeof(float));
            }
            ggml_cgraph * graph = ggml_new_graph(ctx);
            if (!graph) {
                ggml_backend_buffer_free(buffer);
                return false;
            }
            ggml_build_forward_expand(graph, projected);
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_sched_reset(scheduler);
            const ggml_status status = ggml_backend_sched_graph_compute(scheduler, graph);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_backend_sched_synchronize(scheduler);
                ggml_backend_buffer_free(buffer);
                return false;
            }
            next_tensors[layer] = projected;
            next_buffers.emplace_back(buffer);
            next_overlay_bytes += static_cast<size_t>(n_embd) * row_count * sizeof(float);
        }
        composed_ctxs = std::move(next_contexts);
        composed_bufs = std::move(next_buffers);
        composed_tensors = std::move(next_tensors);
        composed_row_by_coefficients = std::move(next_row_by_coefficients);
        composed_overlay_key = requested_overlay_key;
        composed_basis_key = requested_key;
        composed_zero_overlay_row = static_cast<int32_t>(unique_coefficients.size());
        composed_overlay_bytes_value = next_overlay_bytes;
    }

    std::vector<int32_t> next_sequence_rows;
    next_sequence_rows.reserve(coefficients.size());
    for (const float * row : coefficients) {
        const std::string row_key(reinterpret_cast<const char *>(row),
            static_cast<size_t>(rank) * sizeof(float));
        const bool zero = std::all_of(row, row + rank, [](float value) { return value == 0.0f; });
        next_sequence_rows.push_back(zero ? composed_zero_overlay_row :
            composed_row_by_coefficients.at(row_key));
    }

    tensors = composed_tensors;
    device_resident_flag = false;
    for (const uint32_t layer : layer_indices) {
        const auto buft = model.select_buft(layer);
        device_resident_flag = device_resident_flag ||
            ggml_backend_dev_type(ggml_backend_buft_get_device(buft)) != GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    this->seq_ids = seq_ids;
    this->seq_overlay_rows = std::move(next_sequence_rows);
    this->zero_overlay_row = composed_zero_overlay_row;
    this->n_embd = n_embd;
    this->layer_start = static_cast<int32_t>(layer_indices.front());
    this->layer_end = static_cast<int32_t>(layer_indices.back());
    this->factorized = false;
    this->factorized_rank = rank;
    this->active = true;
    if (!overlay_cache_hit) {
        last_composition_us_value = static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::microseconds>(std::chrono::steady_clock::now() - compose_started).count());
    }
    return true;
}

bool llama_adapter_cvec_batch::apply_impl(
        const llama_model & model,
        const std::vector<llama_seq_id> & seq_ids,
        const std::vector<const float *> & data,
        const size_t data_len,
        const int32_t n_embd,
        const std::vector<uint32_t> & layer_indices,
        const bool dense_layout) {
    clear();

    if (seq_ids.empty() || seq_ids.size() != data.size() || n_embd <= 0 ||
            layer_indices.empty() ||
            n_embd != static_cast<int32_t>(model.hparams.n_embd)) {
        return false;
    }

    if (std::adjacent_find(layer_indices.begin(), layer_indices.end(),
            [](const uint32_t left, const uint32_t right) { return left >= right; }) !=
            layer_indices.end() || layer_indices.front() < 1 ||
            layer_indices.back() >= model.hparams.n_layer()) {
        return false;
    }

    const size_t layer_count = dense_layout
        ? model.hparams.n_layer() - 1
        : layer_indices.size();
    const size_t required = static_cast<size_t>(n_embd) * layer_count;
    if (data_len < required) {
        return false;
    }
    if (std::any_of(data.begin(), data.end(), [](const float * values) {
            return values == nullptr;
        })) {
        return false;
    }

    std::vector<llama_seq_id> sorted_ids = seq_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        return false;
    }

    if (!init(model, n_embd, seq_ids.size(), layer_indices)) {
        clear();
        return false;
    }

    for (size_t layer_index = 0; layer_index < layer_indices.size(); ++layer_index) {
        const uint32_t layer = layer_indices[layer_index];
        ggml_tensor * tensor = tensors[layer];
        for (size_t row = 0; row < data.size(); ++row) {
            const size_t off = static_cast<size_t>(n_embd) *
                (dense_layout ? layer - 1 : layer_index);
            ggml_backend_tensor_set(
                tensor,
                data[row] + off,
                row * static_cast<size_t>(n_embd) * sizeof(float),
                static_cast<size_t>(n_embd) * sizeof(float));
        }
    }

    this->seq_ids = seq_ids;
    this->n_embd = n_embd;
    this->layer_start = static_cast<int32_t>(layer_indices.front());
    this->layer_end = static_cast<int32_t>(layer_indices.back());
    this->active = true;
    return true;
}

bool llama_adapter_cvec_batch::apply(
        const llama_model & model,
        const std::vector<llama_seq_id> & seq_ids,
        const std::vector<const float *> & data,
        const size_t data_len,
        const int32_t n_embd,
        const int32_t il_start,
        const int32_t il_end) {
    if (il_start < 1 || il_end < il_start ||
            static_cast<size_t>(il_end) >= model.hparams.n_layer()) {
        return false;
    }
    std::vector<uint32_t> layer_indices;
    layer_indices.reserve(static_cast<size_t>(il_end - il_start + 1));
    for (int32_t layer = il_start; layer <= il_end; ++layer) {
        layer_indices.push_back(static_cast<uint32_t>(layer));
    }
    return apply_impl(model, seq_ids, data, data_len, n_embd, layer_indices, true);
}

bool llama_adapter_cvec_batch::apply_sparse(
        const llama_model & model,
        const std::vector<llama_seq_id> & seq_ids,
        const std::vector<const float *> & data,
        const size_t data_len,
        const int32_t n_embd,
        const std::vector<uint32_t> & layer_indices) {
    return apply_impl(model, seq_ids, data, data_len, n_embd, layer_indices, false);
}

int32_t llama_adapter_cvec_batch::row_for_sequence(const llama_seq_id sequence_id) const {
    const auto it = std::find(seq_ids.begin(), seq_ids.end(), sequence_id);
    if (it == seq_ids.end()) return -1;
    const size_t index = static_cast<size_t>(it - seq_ids.begin());
    return seq_overlay_rows.size() == seq_ids.size()
        ? seq_overlay_rows[index]
        : static_cast<int32_t>(index);
}

int32_t llama_adapter_cvec_batch::row_for_token(
        const llama_ubatch & ubatch,
        const uint32_t token_index) const {
    int32_t row = -1;
    bool contains_unbound_sequence = false;
    for (int32_t s = 0; s < ubatch.n_seq_id[token_index]; ++s) {
        const llama_seq_id sequence_id = ubatch.seq_id[token_index][s];
        const int32_t candidate = row_for_sequence(sequence_id);
        if (candidate < 0) {
            contains_unbound_sequence = true;
            continue;
        }

        if (row != -1 && row != candidate) {
            // A token shared by multiple sequences cannot receive two
            // different overlays.  The server must split such a batch or
            // provide one shared row instead of silently choosing seq_id[0].
            return -1;
        }
        row = candidate;
    }
    // A token shared with an unbound sequence is not safe to steer: use the
    // extra zero row. A wholly unbound token uses that same row. The server
    // never silently selects another request's cvec.
    return contains_unbound_sequence || row < 0
        ? (zero_overlay_row >= 0 ? zero_overlay_row : static_cast<int32_t>(seq_ids.size()))
        : row;
}

ggml_tensor * llama_adapter_cvec_batch::apply_callback(
        ggml_context * ctx,
        ggml_tensor * cur,
        const int il,
        const llama_ubatch & ubatch,
        void * user_data) {
    auto * batch = static_cast<llama_adapter_cvec_batch *>(user_data);
    ggml_tensor * table = batch->tensor_for(il);
    ggml_tensor * basis = batch->factorized && static_cast<size_t>(il) < batch->factorized_basis_tensors.size()
        ? batch->factorized_basis_tensors[il] : nullptr;
    ggml_tensor * coefficients = batch->factorized && static_cast<size_t>(il) < batch->factorized_coefficient_tensors.size()
        ? batch->factorized_coefficient_tensors[il] : nullptr;
    if ((table == nullptr && (basis == nullptr || coefficients == nullptr)) || ubatch.n_tokens == 0) {
        return cur;
    }
    if (batch->callback_failed_flag.load()) {
        return cur;
    }

    if (cur->ne[2] != 1 || cur->ne[3] != 1 || cur->ne[1] <= 0) {
        LLAMA_LOG_ERROR(
            "%s: unsupported batched cvec graph shape [%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
            __func__, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
        batch->mark_callback_failure("unsupported cvec graph tensor shape");
        return cur;
    }

    const ggml_tensor * storage = table ? table : basis;
    if (storage->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: batch cvec table has no backend buffer\n", __func__);
        batch->mark_callback_failure("cvec graph table has no backend buffer");
        return cur;
    }

    const uint32_t index_count = static_cast<uint32_t>(cur->ne[1]);

    std::lock_guard<std::mutex> lock(batch->selector_mutex);
    auto & selector = batch->selector_states[ctx];
    if (selector.tensor == nullptr || selector.count != index_count) {
        selector.tensor = nullptr;
        selector.count = 0;
        selector.buf.reset();
        selector.ctx.reset();

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        selector.ctx.reset(ggml_init(params));
        if (!selector.ctx) {
            batch->mark_callback_failure("could not allocate cvec selector context");
            return cur;
        }
        selector.tensor = ggml_new_tensor_1d(
            selector.ctx.get(), GGML_TYPE_I32, index_count);
        if (selector.tensor == nullptr) {
            selector.ctx.reset();
            batch->mark_callback_failure("could not allocate cvec selector tensor");
            return cur;
        }
        selector.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            selector.ctx.get(), ggml_backend_buffer_get_type(storage->buffer)));
        if (!selector.buf) {
            selector.tensor = nullptr;
            selector.ctx.reset();
            batch->mark_callback_failure("could not allocate cvec selector backend buffer");
            return cur;
        }
        selector.count = index_count;
    }

    std::vector<int32_t> indices(index_count, -1);
    for (uint32_t i = 0; i < index_count; ++i) {
        int32_t row = -1;
        if (index_count == ubatch.n_tokens) {
            row = batch->row_for_token(ubatch, i);
        } else if (index_count == ubatch.n_seqs_unq && ubatch.seq_id_unq != nullptr) {
            row = batch->row_for_sequence(ubatch.seq_id_unq[i]);
            if (row < 0) row = batch->zero_overlay_row >= 0
                ? batch->zero_overlay_row : static_cast<int32_t>(batch->seq_ids.size());
        }
        if (row < 0) {
            LLAMA_LOG_ERROR(
                "%s: cannot map graph indices to cvec batch (index_count=%u tokens=%u seqs=%u)\n",
                __func__, index_count, ubatch.n_tokens, ubatch.n_seqs_unq);
            batch->mark_callback_failure("could not map graph tokens to cvec sequence rows");
            return cur;
        }
        indices[i] = row;
    }
    ggml_backend_tensor_set(
        selector.tensor, indices.data(), 0, indices.size() * sizeof(int32_t));
    if (ggml_backend_buffer_get_type(storage->buffer) !=
            ggml_backend_buffer_get_type(selector.buf.get())) {
        LLAMA_LOG_ERROR("%s: selector and cvec table use incompatible backends\n", __func__);
        batch->mark_callback_failure("selector and cvec table use incompatible backends");
        return cur;
    }

    if (batch->factorized) {
        ggml_tensor * coefficient_rows = ggml_get_rows(ctx, coefficients, selector.tensor);
        if (coefficient_rows == nullptr) {
            batch->mark_callback_failure("could not select factorized cvec coefficient rows");
            return cur;
        }
        ggml_tensor * projected = ggml_mul_mat(ctx, basis, coefficient_rows);
        if (projected == nullptr) {
            batch->mark_callback_failure("could not build factorized cvec matrix product");
            return cur;
        }
        ggml_tensor * result = ggml_add(ctx, cur, projected);
        if (result == nullptr) {
            batch->mark_callback_failure("could not build factorized cvec graph operation");
            return cur;
        }
        batch->callback_applied_flag.store(true);
        return result;
    }
    ggml_tensor * result = ggml_add(ctx, cur, ggml_get_rows(ctx, table, selector.tensor));
    if (result == nullptr) {
        batch->mark_callback_failure("could not build sparse cvec graph operation");
        return cur;
    }
    batch->callback_applied_flag.store(true);
    return result;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static void llama_adapter_lora_init_impl(llama_model & model, FILE * file, llama_adapter_lora & adapter) {
    ggml_context * ctx_init;
    gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file_ptr(file, meta_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter from file");
    }

    ggml_context_ptr ctx { ctx_init };

    // must come after gguf_init_from_file_ptr, the llama_file constructor moves the file position
    llama_file gguf_file(file);

    // check metadata
    {
        const gguf_context * gguf_ctx = ctx_gguf.get();

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const std::string value = gguf_kv_to_str(gguf_ctx, i);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            const size_t MAX_VALUE_LEN = 40;
            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(ctx_gguf.get(), kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
            const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        const auto * model_tensor = model.get_tensor(name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }
        // the LoRA delta of a Hadamard-folded weight reads the transformed input, so the result is wrong
        if (model.hdmd.weight_blocks.count(name) || model.hdmd.inverse_blocks.count(name)) {
            throw std::runtime_error("LoRA tensor '" + name + "' targets a prism.hadamard folded weight, which is not supported");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            const size_t offs = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), gguf_find_tensor(ctx_gguf.get(), orig->name));
            const size_t size = ggml_nbytes(orig);
            if (offs + size < offs || offs + size > gguf_file.size()) {
                throw std::runtime_error(format("LoRA tensor '%s' data is not within the file bounds, file is corrupted or incomplete", orig->name));
            }
            read_buf.resize(size);
            gguf_file.seek(offs, SEEK_SET);
            gguf_file.read_raw(read_buf.data(), size);
            ggml_backend_tensor_set(dev, read_buf.data(), 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    // register adapter with model
    model.loras.insert(&adapter);

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    FILE * file = ggml_fopen(path_lora, "rb");
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path_lora, strerror(errno));
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    fclose(file);

    return adapter;
}

llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(llama_model * model, FILE * file) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }

    llama_adapter_lora * adapter = new llama_adapter_lora(model);

    try {
        llama_adapter_lora_init_impl(*model, file, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return;
    }

    if (adapter->model != nullptr) {
        adapter->model->loras.erase(adapter);
        adapter->model = nullptr;
    }

    delete adapter;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}
