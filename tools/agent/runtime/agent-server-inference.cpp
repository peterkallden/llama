#include "../cli/agent-cli-inference.h"
#include "../runtime/agent-server-generation.h"

#include "agent/agent-prepared-generation.h"
#include "agent/adaptation/flydelta/flydelta-capture.h"
#include "server-context.h"
#include "server-task.h"

#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <utility>

namespace {

struct schema_stream_state {
    std::string content;
    std::string first_valid_json;
    int decoded_tokens = 0;
    int first_valid_decoded_tokens = 0;
    server_task_result_ptr final_response;
    common_agent_generation_stop_reason final_stop_reason = common_agent_generation_stop_reason::error;
};

bool resident_trace_enabled() {
    const char * value = std::getenv("LLAMA_AGENT_RESIDENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void resident_trace(const char * event, const common_agent_generation_request & request, const char * detail = "") {
    if (!resident_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "agent resident trace: event=%s purpose=%s schema=%s n_predict=%d %s\n",
        event,
        common_agent_generation_purpose_name(request.purpose),
        request.json_schema.empty() ? "no" : "yes",
        request.options.n_predict,
        detail);
    std::fflush(stderr);
}

common_agent_generation_stop_reason map_server_stop_reason_string(const std::string & stop) {
    if (stop == "eos") {
        return common_agent_generation_stop_reason::eos;
    }
    if (stop == "limit" || stop == "word") {
        return common_agent_generation_stop_reason::limit;
    }
    if (stop == "none") {
        return common_agent_generation_stop_reason::none;
    }
    return common_agent_generation_stop_reason::error;
}

bool extract_completion_json(
        server_task_result * response,
        std::string & content,
        int & decoded_tokens,
        common_agent_generation_stop_reason & stop_reason) {
    if (response == nullptr) {
        return false;
    }
    const auto payload = response->to_json();
    if (!payload.is_object() || !payload.contains("content") || !payload["content"].is_string()) {
        return false;
    }
    content = payload["content"].get<std::string>();
    decoded_tokens = payload.value("tokens_predicted", decoded_tokens);
    stop_reason = map_server_stop_reason_string(payload.value("stop_type", std::string{"none"}));
    return true;
}

void apply_server_success(
        common_agent_generation_result & result,
        std::string content,
        int decoded_tokens,
        common_agent_generation_stop_reason stop_reason) {
    result.content = std::move(content);
    result.decoded_tokens = decoded_tokens;
    result.status = common_agent_generation_status::completed;
    result.stop_reason = stop_reason;
    result.error_message.clear();
}

void apply_server_error(
        server_task_result * response,
        common_agent_generation_result & result,
        common_agent_generation_status status = common_agent_generation_status::errored,
        common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::error,
        std::string fallback_error = {}) {
    result.status = status;
    result.stop_reason = stop_reason;
    if (response != nullptr && response->is_error()) {
        const auto * error = static_cast<const server_task_result_error *>(response);
        result.error_message = error->err_msg;
    } else {
        result.error_message = std::move(fallback_error);
    }
}

void apply_server_capture(
        const server_task_result_cmpl_final & response,
        const common_agent_generation_request & request,
        common_agent_generation_result & result) {
    if (!request.flydelta_capture || !request.flydelta_capture->enabled ||
            !response.capture.attempted) {
        return;
    }
    auto capture = std::make_shared<common_flydelta_hidden_state_capture>();
    capture->model_profile_fingerprint = response.capture.model_profile_fingerprint;
    capture->capture_layout_revision = response.capture.capture_layout_revision;
    capture->layer_indices = response.capture.layer_indices;
    capture->position = static_cast<common_flydelta_capture_position>(response.capture.position);
    capture->token_index = response.capture.token_index;
    capture->n_embd = response.capture.n_embd;
    capture->values = response.capture.values;
    capture->captured = response.capture.captured;
    capture->failure_reason = response.capture.failure_reason;
    result.flydelta_capture = std::move(capture);
}

bool record_schema_partial(
        const common_agent_generation_request & request,
        server_task_result * response,
        schema_stream_state & state,
        common_agent_generation_result & result) {
    std::string delta;
    common_agent_generation_stop_reason ignored_stop_reason = common_agent_generation_stop_reason::none;
    int partial_decoded_tokens = state.decoded_tokens;
    if (!extract_completion_json(response, delta, partial_decoded_tokens, ignored_stop_reason)) {
        return false;
    }
    state.content += delta;
    state.decoded_tokens = partial_decoded_tokens;
    const auto parsed = nlohmann::ordered_json::parse(state.content, nullptr, false);
    if (!parsed.is_discarded() && state.first_valid_json.empty()) {
        state.first_valid_json = state.content;
        state.first_valid_decoded_tokens = state.decoded_tokens;
        resident_trace("schema-first-valid", request);
        apply_server_success(result, std::move(state.first_valid_json), state.first_valid_decoded_tokens, common_agent_generation_stop_reason::json_schema);
        return true;
    }
    return false;
}

bool apply_schema_final_response(
        const common_agent_generation_request & request,
        schema_stream_state & state,
        common_agent_generation_result & result) {
    if (state.final_response == nullptr) {
        return false;
    }
    if (!extract_completion_json(state.final_response.get(), state.content, state.decoded_tokens, state.final_stop_reason)) {
        apply_server_error(nullptr, result, common_agent_generation_status::errored, common_agent_generation_stop_reason::error,
            "server returned a non-completion final result");
        resident_trace("schema-invalid-final-type", request);
        return true;
    }

    const auto parsed = nlohmann::ordered_json::parse(state.content, nullptr, false);
    if (!parsed.is_discarded()) {
        resident_trace("schema-final-valid", request);
        apply_server_success(result, state.content, state.decoded_tokens, common_agent_generation_stop_reason::json_schema);
        return true;
    }

    return false;
}

bool apply_reasoning_schema_fallback(
        const common_agent_generation_request & request,
        schema_stream_state & state,
        common_agent_generation_result & result) {
    if (request.purpose != common_agent_generation_purpose::reasoning) {
        return false;
    }
    if (state.content.empty()) {
        state.content = R"({"summary":"Reasoning step reached the generation limit without structured output.","format":"unstructured"})";
    }
    apply_server_success(result, std::move(state.content), state.decoded_tokens, state.final_stop_reason);
    resident_trace("schema-reasoning-unstructured", request);
    return true;
}

class server_context_agent_inference final : public common_agent_inference {
public:
    server_context_agent_inference(
            server_context & server,
            const common_params & params_base,
            const std::vector<llama_logit_bias> & logit_bias_eog,
            const common_chat_templates * templates,
            bool supports_image,
            bool supports_audio)
        : server(server), params_base(params_base), logit_bias_eog(logit_bias_eog), templates(templates),
          supports_image(supports_image), supports_audio(supports_audio) {}

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        result = {};

        try {
            resident_trace("start", request);
            std::string flydelta_error;
            if (!server_context_agent_generation_supports_flydelta(request, flydelta_error)) {
                result.error_message = flydelta_error;
                resident_trace("flydelta-unsupported", request, flydelta_error.c_str());
                return false;
            }
            common_agent_prepared_generation prepared;
            common_chat_params chat_params;
            if (!common_agent_prepare_chat_generation(templates, request, prepared, &chat_params)) {
                result.error_message = "failed to prepare server generation";
                return false;
            }
            result.chat_params = chat_params;
            if (resident_trace_enabled()) {
                std::fprintf(stderr, "agent resident trace: event=prepared purpose=%s stream=%s prompt_bytes=%zu\n",
                    common_agent_generation_purpose_name(request.purpose),
                    prepared.stream ? "yes" : "no",
                    prepared.prompt.size());
                std::fflush(stderr);
            }

            server_response_reader reader = server.get_response_reader();
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.id = reader.get_new_id();
            task.cli = true;
            task.cli_prompt = prepared.prompt;
            for (const auto & resource : request.input_resources) {
                const auto mime_type = common_normalize_resource_media_type(resource.resource.mime_type);
                const bool is_image = mime_type.rfind("image/", 0) == 0;
                const bool is_audio = mime_type.rfind("audio/", 0) == 0;
                if (!is_image && !is_audio) {
                    continue;
                }
                const bool native_supported = is_image
                    ? supports_image
                    : supports_audio;
                if (!native_supported) {
                    if (is_image && resource.read_text_fallback) {
                        std::string text;
                        std::string fallback_error;
                        const size_t max_text_bytes = resource.resource.size_bytes > 0
                            ? resource.resource.size_bytes
                            : 16 * 1024 * 1024;
                        if (resource.read_text_fallback(max_text_bytes, text, fallback_error)) {
                            task.cli_prompt += "\n[Extracted image text]\n" + text;
                            continue;
                        }
                        if (resource.required) {
                            result.error_message = "image text fallback failed for " + resource.resource.uri;
                            if (!fallback_error.empty()) {
                                result.error_message += ": " + fallback_error;
                            }
                            return false;
                        }
                        continue;
                    }
                    if (resource.required) {
                        result.error_message = (is_image ? "required image" : "required audio") +
                            std::string(" resource is unsupported by the loaded model: ") + resource.resource.uri;
                        return false;
                    }
                    continue;
                }
                if (!resource.read_bytes) {
                    if (resource.required) {
                        result.error_message = "required media resource has no host resolver: " + resource.resource.uri;
                        return false;
                    }
                    continue;
                }
                std::string bytes;
                std::string read_error;
                const size_t max_bytes = resource.resource.size_bytes > 0
                    ? resource.resource.size_bytes
                    : 64 * 1024 * 1024;
                if (!resource.read_bytes(max_bytes, bytes, read_error)) {
                    result.error_message = "failed to read image resource " + resource.resource.uri;
                    if (!read_error.empty()) {
                        result.error_message += ": " + read_error;
                    }
                    return false;
                }
                task.cli_prompt += "\n" + std::string(get_media_marker());
                task.cli_files.emplace_back(bytes.begin(), bytes.end());
            }
            task.params = make_server_task_params_from_prepared_generation(
                params_base,
                request,
                prepared,
                logit_bias_eog);
            reader.post_task(std::move(task));

            if (prepared.stream) {
                schema_stream_state state;

                while (auto response = reader.next([]() { return false; })) {
                    if (response->is_error()) {
                        resident_trace("error", request);
                        apply_server_error(response.get(), result);
                        return false;
                    }
                    if (!response->is_stop()) {
                        if (!record_schema_partial(request, response.get(), state, result)) {
                            continue;
                        }
                        reader.stop();
                        return true;
                    }
                    if (response->is_stop()) {
                        state.final_response = std::move(response);
                        resident_trace("stop", request);
                        break;
                    }
                }

                if (apply_schema_final_response(request, state, result)) {
                    return common_agent_generation_succeeded(result);
                }
                if (!state.first_valid_json.empty()) {
                    resident_trace("schema-first-valid-return", request);
                    apply_server_success(result, std::move(state.first_valid_json), state.first_valid_decoded_tokens, common_agent_generation_stop_reason::json_schema);
                    return true;
                }
                if (apply_reasoning_schema_fallback(request, state, result)) {
                    return true;
                }

                result.content = std::move(state.content);
                result.decoded_tokens = state.decoded_tokens;
                apply_server_error(
                    state.final_response.get(),
                    result,
                    common_agent_generation_status::errored,
                    state.final_response == nullptr ? common_agent_generation_stop_reason::error : state.final_stop_reason,
                    "generation ended before producing valid JSON for the requested schema");
                resident_trace("schema-invalid", request);
                return false;
            }

            auto responses = reader.wait_for_all([]() { return false; });
            if (responses.is_terminated || responses.error || responses.results.empty()) {
                if (responses.error) {
                    apply_server_error(responses.error.get(), result);
                } else {
                    apply_server_error(
                        nullptr,
                        result,
                        responses.is_terminated ? common_agent_generation_status::cancelled : common_agent_generation_status::errored,
                        responses.is_terminated ? common_agent_generation_stop_reason::cancelled : common_agent_generation_stop_reason::error,
                        responses.is_terminated ? "server task was terminated" : "server task produced no completion result");
                }
                resident_trace("nonstream-error", request);
                return false;
            }

            auto * response = responses.results.front().get();
            std::string content;
            int decoded_tokens = 0;
            common_agent_generation_stop_reason stop_reason = common_agent_generation_stop_reason::none;
            if (!response->is_stop() || !extract_completion_json(response, content, decoded_tokens, stop_reason)) {
                apply_server_error(nullptr, result, common_agent_generation_status::errored, common_agent_generation_stop_reason::error,
                    "server returned a non-completion result");
                return false;
            }

            apply_server_success(result, std::move(content), decoded_tokens, stop_reason);
            if (const auto * final_response = dynamic_cast<const server_task_result_cmpl_final *>(response)) {
                apply_server_capture(*final_response, request, result);
            }
            resident_trace("nonstream-success", request);
            return true;
        } catch (const std::exception & err) {
            apply_server_error(nullptr, result, common_agent_generation_status::errored, common_agent_generation_stop_reason::error, err.what());
            std::fprintf(stderr, "server_context agent inference failed: %s\n", err.what());
            return false;
        }
    }

    bool score_teacher_forced_choice(
            const common_agent_teacher_forced_choice_request & request,
            common_agent_teacher_forced_choice_result & result) override {
        result = {};

        try {
            std::string flydelta_error;
            if (!server_context_agent_generation_supports_flydelta(request.context, flydelta_error)) {
                result.error_message = flydelta_error;
                return false;
            }

            const std::string & positive = request.positive_continuation.empty()
                ? request.positive_choice : request.positive_continuation;
            const std::string & negative = request.negative_continuation.empty()
                ? request.negative_choice : request.negative_continuation;
            if (request.choice_prefix.empty() || positive.empty() || negative.empty()) {
                result.error_message = "server teacher-forced scoring requires a prefix and two continuations";
                return false;
            }
            if (!request.context.input_resources.empty()) {
                result.error_message = "server teacher-forced scoring does not support multimodal resources yet";
                return false;
            }

            common_agent_prepared_generation prepared;
            common_chat_params chat_params;
            if (!common_agent_prepare_chat_generation(
                    templates, request.context, prepared, &chat_params)) {
                result.error_message = "failed to prepare server teacher-forced scoring prompt";
                return false;
            }

            const std::string prefix = prepared.prompt + request.choice_prefix;
            auto make_score_task = [&](server_response_reader & reader, const std::string & continuation) {
                server_task task(SERVER_TASK_TYPE_TEACHER_FORCED_SCORE);
                task.id = reader.get_new_id();
                task.cli = true;
                task.cli_prompt = prefix;
                task.teacher_forced_target = continuation;
                task.params = make_server_task_params_from_prepared_generation(
                    params_base, request.context, prepared, logit_bias_eog);
                // Scoring is a non-streaming, non-sampling operation.  In
                // particular, disable prompt-cache reuse so positive and
                // negative choices are fresh isolated evaluations.
                task.params.stream = false;
                task.params.cache_prompt = false;
                task.params.n_predict = 0;
                return task;
            };

            auto score_one = [&](const std::string & continuation,
                    server_task_result_teacher_score & score) {
                // Use a fresh response reader for each side. The server host
                // normally runs with one slot; sequential tasks keep the
                // positive and negative prompt/KV lifetimes unambiguous.
                server_response_reader reader = server.get_response_reader();
                server_task task = make_score_task(reader, continuation);
                reader.post_task(std::move(task));
                auto responses = reader.wait_for_all([]() { return false; });
                if (responses.is_terminated || responses.error || responses.results.size() != 1) {
                    if (responses.error && responses.error->is_error()) {
                        const auto * error = static_cast<const server_task_result_error *>(responses.error.get());
                        result.error_message = error->err_msg;
                    } else {
                        result.error_message = responses.is_terminated
                            ? "server teacher-forced scoring was terminated"
                            : "server teacher-forced scoring returned an incomplete result";
                    }
                    return false;
                }
                const auto & response = responses.results.front();
                if (response->is_error()) {
                    const auto * error = static_cast<const server_task_result_error *>(response.get());
                    result.error_message = error->err_msg;
                    return false;
                }
                const auto * score_result = dynamic_cast<const server_task_result_teacher_score *>(response.get());
                if (score_result == nullptr) {
                    result.error_message = "server returned a non-scoring result for teacher-forced task";
                    return false;
                }
                score = *score_result;
                return true;
            };

            server_task_result_teacher_score positive_result;
            server_task_result_teacher_score negative_result;
            if (!score_one(positive, positive_result) || !score_one(negative, negative_result)) {
                return false;
            }
            result.available = positive_result.token_count > 0 && negative_result.token_count > 0;
            result.positive_total_logprob = static_cast<float>(positive_result.total_logprob);
            result.negative_total_logprob = static_cast<float>(negative_result.total_logprob);
            result.positive_token_count = positive_result.token_count;
            result.negative_token_count = negative_result.token_count;
            return true;
        } catch (const std::exception & err) {
            result = {};
            result.error_message = err.what();
            std::fprintf(stderr, "server teacher-forced scoring failed: %s\n", err.what());
            return false;
        }
    }

    bool score_teacher_forced_choice_batch(
            const common_agent_teacher_forced_choice_batch_request & request,
            common_agent_teacher_forced_choice_batch_result & result) override {
        result = {};
        if (request.choices.empty() || request.choices.size() > 256) {
            result.error_message = "server teacher-forced scoring batch has an invalid size";
            return false;
        }

        // Submit the complete batch through one server response reader. The
        // server may still execute entries serially when their cvecs differ
        // (the current context-wide cvec contract requires that), but this
        // keeps task construction, response collection and cancellation at a
        // single backend boundary. It is therefore a real backend batch
        // implementation without pretending that per-sequence overlays are
        // already supported by the server graph.
        struct pending_choice {
            size_t result_index = 0;
        };
        std::vector<server_task> tasks;
        tasks.reserve(request.choices.size() * 2);
        std::vector<pending_choice> pending;
        pending.reserve(request.choices.size());

        for (const auto & choice_request : request.choices) {
            const std::string & positive = choice_request.positive_continuation.empty()
                ? choice_request.positive_choice : choice_request.positive_continuation;
            const std::string & negative = choice_request.negative_continuation.empty()
                ? choice_request.negative_choice : choice_request.negative_continuation;
            if (choice_request.choice_prefix.empty() || positive.empty() || negative.empty()) {
                result.error_message =
                    "server teacher-forced scoring requires a prefix and two continuations";
                return false;
            }
            if (!choice_request.context.input_resources.empty()) {
                result.error_message =
                    "server teacher-forced scoring does not support multimodal resources yet";
                return false;
            }

            common_agent_prepared_generation prepared;
            common_chat_params chat_params;
            if (!common_agent_prepare_chat_generation(
                    templates, choice_request.context, prepared, &chat_params)) {
                result.error_message =
                    "failed to prepare server teacher-forced scoring prompt";
                return false;
            }

            const std::string prefix = prepared.prompt + choice_request.choice_prefix;
            auto make_score_task = [&](const std::string & continuation) {
                server_task task(SERVER_TASK_TYPE_TEACHER_FORCED_SCORE);
                task.id = 0; // assigned by server_response_reader::post_tasks
                task.cli = true;
                task.cli_prompt = prefix;
                task.teacher_forced_target = continuation;
                task.params = make_server_task_params_from_prepared_generation(
                    params_base, choice_request.context, prepared, logit_bias_eog);
                // Every task is a fresh isolated comparison. In particular,
                // do not let one arm reuse prompt/KV state produced by a
                // different cvec or continuation.
                task.params.stream = false;
                task.params.cache_prompt = false;
                task.params.n_predict = 0;
                return task;
            };

            pending.push_back({tasks.size()});
            tasks.push_back(make_score_task(positive));
            tasks.push_back(make_score_task(negative));
        }

        server_response_reader reader = server.get_response_reader();
        for (auto & task : tasks) {
            task.id = reader.get_new_id();
        }
        reader.post_tasks(std::move(tasks));
        const auto responses = reader.wait_for_all([]() { return false; });
        if (responses.is_terminated || responses.error) {
            if (responses.error && responses.error->is_error()) {
                const auto * error =
                    static_cast<const server_task_result_error *>(responses.error.get());
                result.error_message = error->err_msg;
            } else {
                result.error_message = responses.is_terminated
                    ? "server teacher-forced scoring batch was terminated"
                    : "server teacher-forced scoring batch returned an error";
            }
            return false;
        }

        result.choices.reserve(request.choices.size());
        for (size_t choice_index = 0; choice_index < pending.size(); ++choice_index) {
            const size_t positive_index = pending[choice_index].result_index;
            const size_t negative_index = positive_index + 1;
            if (negative_index >= responses.results.size() ||
                    responses.results[positive_index] == nullptr ||
                    responses.results[negative_index] == nullptr) {
                result.error_message =
                    "server teacher-forced scoring batch returned incomplete results";
                result.choices.clear();
                return false;
            }

            const auto * positive_result = dynamic_cast<const server_task_result_teacher_score *>(
                responses.results[positive_index].get());
            const auto * negative_result = dynamic_cast<const server_task_result_teacher_score *>(
                responses.results[negative_index].get());
            if (positive_result == nullptr || negative_result == nullptr) {
                result.error_message =
                    "server returned a non-scoring result for teacher-forced batch task";
                result.choices.clear();
                return false;
            }

            common_agent_teacher_forced_choice_result choice_result;
            choice_result.available = positive_result->token_count > 0 &&
                negative_result->token_count > 0;
            choice_result.positive_total_logprob =
                static_cast<float>(positive_result->total_logprob);
            choice_result.negative_total_logprob =
                static_cast<float>(negative_result->total_logprob);
            choice_result.positive_token_count = positive_result->token_count;
            choice_result.negative_token_count = negative_result->token_count;
            result.choices.push_back(std::move(choice_result));
        }
        return true;
    }

private:
    server_context & server;
    common_params params_base;
    std::vector<llama_logit_bias> logit_bias_eog;
    const common_chat_templates * templates;
    bool supports_image = false;
    bool supports_audio = false;
};

} // namespace

std::unique_ptr<common_agent_inference> make_server_context_agent_inference(
    server_context & server,
    const common_params & params_base,
    const std::vector<llama_logit_bias> & logit_bias_eog,
    const common_chat_templates * templates,
    bool supports_image,
    bool supports_audio) {
    return std::make_unique<server_context_agent_inference>(
        server, params_base, logit_bias_eog, templates, supports_image, supports_audio);
}
