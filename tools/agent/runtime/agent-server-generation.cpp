#include "agent-server-generation.h"

#include "agent/adaptation/flydelta/flydelta-activation.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"

#include "hash/hash.h"

#include <cstring>

namespace {

server_task_cvec_ptr make_server_task_cvec(
        const common_flydelta_activation_result & activation) {
    const auto & overlay = activation.overlay;
    auto cvec = std::make_shared<server_task_cvec>();
    cvec->identity = overlay.artifact_id;
    cvec->n_embd = overlay.n_embd;
    cvec->il_start = overlay.il_start;
    cvec->il_end = overlay.il_end;
    cvec->data = overlay.data;
    for (float & value : cvec->data) {
        value *= overlay.scale;
    }
    if (activation.sparse_overlay.enabled) {
        cvec->sparse_layer_indices = activation.sparse_overlay.layer_indices;
        cvec->sparse_data = activation.sparse_overlay.data;
    }

    std::string fingerprint;
    fingerprint.reserve(sizeof(overlay.n_embd) + sizeof(overlay.il_start) +
        sizeof(overlay.il_end) + sizeof(overlay.scale) +
        overlay.data.size() * sizeof(float));
    fingerprint.append(reinterpret_cast<const char *>(&overlay.n_embd), sizeof(overlay.n_embd));
    fingerprint.append(reinterpret_cast<const char *>(&overlay.il_start), sizeof(overlay.il_start));
    fingerprint.append(reinterpret_cast<const char *>(&overlay.il_end), sizeof(overlay.il_end));
    // The server receives the already scaled cvec. Keep the source scale in
    // the fingerprint as an additional audit signal.
    fingerprint.append(reinterpret_cast<const char *>(&overlay.scale), sizeof(overlay.scale));
    if (!overlay.data.empty()) {
        fingerprint.append(reinterpret_cast<const char *>(overlay.data.data()),
            overlay.data.size() * sizeof(float));
    }
    if (!cvec->sparse_layer_indices.empty()) {
        fingerprint.append(reinterpret_cast<const char *>(cvec->sparse_layer_indices.data()),
            cvec->sparse_layer_indices.size() * sizeof(uint32_t));
        fingerprint.append(reinterpret_cast<const char *>(cvec->sparse_data.data()),
            cvec->sparse_data.size() * sizeof(float));
    }
    cvec->content_hash = "sha256:" + hash_sha256_hex(fingerprint.data(), fingerprint.size());
    return cvec;
}

} // namespace

bool server_context_agent_generation_supports_flydelta(
        const common_agent_generation_request & request,
        std::string & error) {
    error.clear();
    if (request.flydelta_activation) {
        const auto & activation = *request.flydelta_activation;
        if (activation.gate.apply != activation.overlay.enabled) {
            error = "server-context received an inconsistent FlyDelta activation gate";
            return false;
        }
        if (activation.overlay.enabled && activation.overlay.data.empty()) {
            error = "server-context received an empty active FlyDelta overlay";
            return false;
        }
    }
    return true;
}

task_params make_server_task_params_from_prepared_generation(
        const common_params & params_base,
        const common_agent_generation_request & request,
        const common_agent_prepared_generation & prepared,
        const std::vector<llama_logit_bias> & logit_bias_eog) {
    task_params params;
    params.sampling = params_base.sampling;
    params.speculative = params_base.speculative;
    params.n_keep = params_base.n_keep;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.cache_prompt = params_base.cache_prompt;
    params.antiprompt = params_base.antiprompt;
    params.stream = prepared.stream;
    params.n_predict = request.options.n_predict;
    if (request.options.t_max_prompt_ms) {
        params.t_max_prompt_ms = *request.options.t_max_prompt_ms;
    }
    if (request.options.t_max_predict_ms) {
        params.t_max_predict_ms = *request.options.t_max_predict_ms;
    }
    params.sampling.temp = 0.0f;
    params.sampling.grammar = prepared.grammar;
    params.sampling.grammar_lazy = prepared.grammar_lazy;
    params.sampling.grammar_triggers = prepared.grammar_triggers;
    params.sampling.generation_prompt = prepared.generation_prompt;
    params.sampling.ignore_eos = prepared.ignore_eos;
    if (prepared.suppress_eog) {
        params.sampling.logit_bias = logit_bias_eog;
    }
    params.chat_parser_params.format = prepared.chat_format;
    params.chat_parser_params.generation_prompt = prepared.parser_generation_prompt;
    params.chat_parser_params.parse_tool_calls = prepared.parse_tool_calls;

    if (request.flydelta_activation && request.flydelta_activation->overlay.enabled) {
        params.cvec = make_server_task_cvec(*request.flydelta_activation);
        // The current server cvec is context-wide. Until the backend has
        // per-sequence overlay parameters in its graph, prompt/KV reuse must
        // not cross an active FlyDelta intervention boundary.
        params.cache_prompt = false;
        params.n_cache_reuse = 0;
    }

    if (request.flydelta_capture && request.flydelta_capture->enabled) {
        auto capture = std::make_shared<server_task_capture_request>();
        capture->enabled = true;
        capture->layer_indices = request.flydelta_capture->layer_indices;
        capture->token_index = request.flydelta_capture->token_index;
        capture->position = static_cast<int32_t>(request.flydelta_capture->position);
        capture->max_bytes = request.flydelta_capture->max_bytes;
        capture->model_profile_fingerprint = request.flydelta_capture->model_profile_fingerprint;
        capture->capture_layout_revision = request.flydelta_capture->capture_layout_revision;
        params.capture = std::move(capture);
    }

    if (!prepared.parser.empty()) {
        params.chat_parser_params.parser.load(prepared.parser);
    }

    return params;
}
