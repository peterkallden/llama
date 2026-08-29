#include "agent-runtime-session-host.h"

#include "../runtime/agent-runtime-resident.h"
#include "../tooling/agent-tool-provider.h"
#include "memory/memory-context.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

bool session_host_trace_enabled() {
    const char * value = std::getenv("LLAMA_AGENT_RESIDENT_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void session_host_trace(const char * event, const common_agent_runtime_session_host_turn_request & request) {
    if (!session_host_trace_enabled()) {
        return;
    }
    std::fprintf(stderr,
        "agent session host trace: event=%s mode=%s session=%s turn=%s n_predict=%d\n",
        event,
        request.mode == common_agent_runtime_host_mode::chat ? "chat" : "agent",
        request.session_id.c_str(),
        request.turn_id.c_str(),
        request.n_predict);
    std::fflush(stderr);
}

common_agent_failure_class classify_execution_control_failure(
        const common_agent_runtime_execution_control & execution_control) {
    if (execution_control.is_deadline_exceeded()) {
        return common_agent_failure_class::timeout;
    }
    return common_agent_failure_class::execution;
}

common_agent_failure_class classify_turn_failure(
        const common_agent_result & agent_result,
        const common_agent_runtime_execution_control & execution_control) {
    if (execution_control.should_stop()) {
        return classify_execution_control_failure(execution_control);
    }
    if (agent_result.response_generation_status == common_agent_generation_status::cancelled ||
            agent_result.response_stop_reason == common_agent_generation_stop_reason::cancelled) {
        return common_agent_failure_class::execution;
    }
    if (agent_result.response_stop_reason == common_agent_generation_stop_reason::limit) {
        return common_agent_failure_class::limit;
    }
    if (agent_result.response_stop_reason == common_agent_generation_stop_reason::json_schema) {
        return common_agent_failure_class::model_format;
    }
    if (!agent_result.failures.empty()) {
        return agent_result.failures.back().classification;
    }
    return common_agent_failure_class::execution;
}

bool validate_session_host_turn_request(
        const common_agent_runtime_session_host_turn_request & request,
        std::string & error) {
    if (request.prompt.empty()) {
        error = "session host turn request requires a prompt";
        return false;
    }
    if (request.session_id.empty()) {
        error = "session host turn request requires a session_id";
        return false;
    }
    if (request.namespace_id.empty()) {
        error = "session host turn request requires a namespace_id";
        return false;
    }

    const bool project_scope_requested =
        request.memory_scope == common_memory_scope::project ||
        request.plan_scope == common_plan_scope::project;
    if (project_scope_requested && request.project_id.empty()) {
        error = "project-scoped session host turn request requires a project_id";
        return false;
    }

    const bool turn_scope_requested =
        request.memory_scope == common_memory_scope::turn ||
        request.plan_scope == common_plan_scope::turn;
    if (turn_scope_requested && request.turn_id.empty()) {
        error = "turn-scoped session host turn request requires a turn_id";
        return false;
    }

    error.clear();
    return true;
}

} // namespace

common_agent_runtime_session_host_config make_agent_runtime_session_host_config(
        common_agent_runtime_session_host_build_config config) {
    return {
        config.memory_store,
        config.plan_store,
        std::move(config.resident_request),
        std::move(config.policy),
        std::move(config.runtime_config),
        std::move(config.orchestration_config),
        config.memory_scope,
        config.memory_enabled,
        std::move(config.installed_blueprint_candidates),
        std::move(config.tooling),
        std::move(config.tooling_resolver),
        std::move(config.model_residency),
    };
}

common_agent_runtime_session_host::common_agent_runtime_session_host(
        common_agent_runtime_session_host_config config)
    : config(std::move(config)) {}

common_agent_runtime_session_host::~common_agent_runtime_session_host() = default;

common_agent_runtime_session_host_runtime_key common_agent_runtime_session_host::make_runtime_key(
        const common_agent_runtime_session_host_turn_request & request) const {
    return {
        request.session_id,
        request.namespace_id,
        request.project_id,
        request.memory_scope,
        request.plan_scope,
        request.model_profile_id,
    };
}

common_agent_runtime_turn_request common_agent_runtime_session_host::make_base_turn_request(
        const common_agent_runtime_session_host_turn_request & request) const {
    auto resident_request = config.resident_request;
    resident_request.prompt = request.prompt;
    resident_request.session_id = request.session_id;
    resident_request.namespace_id = request.namespace_id;
    resident_request.project_id = request.project_id;
    resident_request.policy_pack = resolve_policy_pack(request);
    resident_request.memory_scope = request.memory_scope;
    resident_request.plan_scope = request.plan_scope;
    if (request.n_predict > 0) {
        resident_request.n_predict = request.n_predict;
    }

    auto turn_request = make_agent_runtime_resident_base_turn_request(resident_request);
    turn_request.policy = config.policy;
    turn_request.policy.agent_inference_backend = resident_request.inference_backend;
    if (request.deliberation_policy_override.has_value()) {
        turn_request.policy.deliberation_policy = *request.deliberation_policy_override;
        turn_request.request.deliberation_policy = *request.deliberation_policy_override;
        // Keep the legacy driver limits aligned with the typed policy. The
        // driver still consumes these compatibility fields while the typed
        // deliberation policy controls the selected mode.
        turn_request.policy.max_reflection_rounds = std::max(
            turn_request.policy.max_reflection_rounds,
            static_cast<size_t>(std::max(0, request.deliberation_policy_override->max_reflection_rounds)));
        turn_request.policy.max_iterations = std::max(
            turn_request.policy.max_iterations,
            static_cast<size_t>(1 + std::max(0, request.deliberation_policy_override->max_plan_revisions)));
        turn_request.policy.max_tool_rounds = std::max(
            turn_request.policy.max_tool_rounds,
            static_cast<size_t>(std::max(0, request.deliberation_policy_override->max_tool_rounds)));
    }
    turn_request.request.input_resources = request.input_resources;
    turn_request.request.event_sink = request.event_sink;
    turn_request.execution_control = request.execution_control;
    turn_request.request.research_should_stop = [control = request.execution_control]() {
        return control.should_stop();
    };
    turn_request.request.research_stop_reason = [control = request.execution_control]() {
        return control.is_deadline_exceeded()
            ? common_agent_research_stop_reason::deadline_exceeded
            : common_agent_research_stop_reason::cancelled;
    };
    turn_request.runtime_config = config.runtime_config;
    turn_request.orchestration_config = config.orchestration_config;
    turn_request.orchestration_config.prompt = request.prompt;
    turn_request.memory_scope = request.memory_scope;
    turn_request.memory_enabled = config.memory_enabled;
    if (turn_request.generation_options.n_predict == 0) {
        turn_request.generation_options.n_predict = resident_request.n_predict;
    }
    if (!turn_request.generation_options.t_max_predict_ms &&
            request.execution_control.timeout_policy.inference_step_timeout_ms > 0) {
        turn_request.generation_options.t_max_predict_ms =
            request.execution_control.timeout_policy.inference_step_timeout_ms;
    }
    return turn_request;
}

std::optional<common_memory_policy_pack> common_agent_runtime_session_host::resolve_policy_pack(
        const common_agent_runtime_session_host_turn_request & request) const {
    if (request.policy_pack.has_value()) {
        return request.policy_pack;
    }
    if (active_policy_pack.has_value()) {
        return active_policy_pack;
    }
    return config.resident_request.policy_pack;
}

void common_agent_runtime_session_host::update_session_policy_pack(
        const common_agent_runtime_session_host_turn_request & request) {
    if (!request.policy_pack.has_value()) {
        return;
    }
    active_policy_pack = request.policy_pack;
    if (runtime) {
        runtime->set_policy_pack(active_policy_pack);
    }
}

void common_agent_runtime_session_host::compact_session_policy_pack_after_reflection(
        const common_agent_result & agent_result,
        common_agent_runtime_session_host_turn_result & result) {
    if (!agent_result.reflected) {
        return;
    }

    const auto resolved_policy_pack = active_policy_pack.has_value()
        ? active_policy_pack
        : config.resident_request.policy_pack;
    if (!resolved_policy_pack.has_value() ||
            !common_memory_policy_pack_needs_compaction(*resolved_policy_pack)) {
        return;
    }

    active_policy_pack = common_memory_compact_policy_pack(*resolved_policy_pack);
    if (runtime) {
        runtime->set_policy_pack(active_policy_pack);
    }

    result.trace.push_back({
        common_runtime_trace_stage::reflection,
        common_runtime_trace_kind::updated,
        "session policy pack compacted after reflection",
        result.plan_id,
        {},
        {},
        {},
        active_policy_pack->id,
    });
    result.trace_count = result.trace.size();
}

bool common_agent_runtime_session_host::resolve_tooling(
        const common_agent_runtime_resident_runtime * runtime,
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_tooling & tooling,
        std::string & error) const {
    tooling = {};
    if (config.tooling_resolver) {
        return config.tooling_resolver(runtime, request, tooling, error);
    }

    tooling = config.tooling;
    error.clear();
    return true;
}

bool common_agent_runtime_session_host::ensure_runtime(
        const common_agent_runtime_session_host_turn_request & request,
        bool & reused,
        std::string & error) {
    const auto requested_key = make_runtime_key(request);
    const bool needs_new_runtime =
        !runtime ||
        !(active_runtime_key == requested_key);

    if (!needs_new_runtime) {
        reused = true;
        error.clear();
        return true;
    }

    reused = false;
    runtime = std::make_unique<common_agent_runtime_resident_runtime>(
        make_agent_runtime_resident_runtime_config(
            config.memory_store,
            &config.plan_store,
            make_base_turn_request(request),
            {},
            config.installed_blueprint_candidates,
            config.tooling,
            config.model_residency,
            request.model_profile_id));
    active_runtime_key = std::move(requested_key);
    error.clear();
    return true;
}

bool common_agent_runtime_session_host::run_turn(
        const common_agent_runtime_session_host_turn_request & request,
        common_agent_runtime_session_host_turn_result & result,
        std::string & error) {
    result = {};
    session_host_trace("enter", request);

    if (!validate_session_host_turn_request(request, error)) {
        result.error = error;
        return false;
    }
    if (request.execution_control.should_stop()) {
        result.cancelled = true;
        result.failure_class = classify_execution_control_failure(request.execution_control);
        result.response_generation_status = common_agent_generation_status::cancelled;
        result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
        result.error = request.execution_control.stop_reason();
        error = result.error;
        return false;
    }

    bool runtime_reused = false;
    if (!ensure_runtime(request, runtime_reused, error)) {
        result.error = error;
        return false;
    }
    session_host_trace("runtime-ready", request);
    update_session_policy_pack(request);
    common_agent_runtime_tooling resolved_tooling;
    if (!resolve_tooling(runtime.get(), request, resolved_tooling, error)) {
        result.error = error;
        return false;
    }
    if (request.execution_control.should_stop()) {
        result.cancelled = true;
        result.error = request.execution_control.stop_reason();
        error = result.error;
        return false;
    }
    runtime->set_tooling(std::move(resolved_tooling));
    runtime->set_execution_control(request.execution_control);

    const std::string turn_id = request.turn_id.empty()
        ? "daemon-turn-" + std::to_string(++generated_turn_counter)
        : request.turn_id;

    common_agent_result agent_result;
    bool ok = false;
    session_host_trace("before-runtime-run", request);
    switch (request.mode) {
        case common_agent_runtime_host_mode::chat:
            ok = runtime->run_chat_prompt(request.prompt, turn_id, request.n_predict, agent_result, error);
            break;
        case common_agent_runtime_host_mode::agent:
            ok = runtime->run_agent_prompt(request.prompt, turn_id, request.n_predict, agent_result, error);
            break;
    }
    session_host_trace("after-runtime-run", request);

    result.ok = ok;
    result.runtime_reused = runtime_reused;
    result.limit_reached = agent_result.limit_reached;
    result.continuation_checkpoint = agent_result.continuation_checkpoint;
    result.reflected = agent_result.reflected;
    result.revised = agent_result.revised;
    result.failure_class = classify_turn_failure(agent_result, request.execution_control);
    result.response = agent_result.response;
    result.plan_id = agent_result.plan_id ? *agent_result.plan_id : "";
    result.total_decoded_tokens = agent_result.total_decoded_tokens;
    result.response_generation_status = agent_result.response_generation_status;
    result.response_stop_reason = agent_result.response_stop_reason;
    if (result.continuation_checkpoint) {
        result.continuation_checkpoint->request_id = request.request_id.empty()
            ? request.turn_id
            : request.request_id;
        result.continuation_checkpoint->turn_id = request.turn_id;
        std::string checkpoint_error;
        if (!common_agent_continuation_checkpoint_valid(
                *result.continuation_checkpoint, checkpoint_error)) {
            result.continuation_checkpoint.reset();
            result.error = "invalid continuation checkpoint: " + checkpoint_error;
            error = result.error;
            return false;
        }
    }
    result.event_count = agent_result.events.size();
    result.trace_count = agent_result.trace.size();
    result.memory_learning_related_count = agent_result.memory_learning_related_count;
    result.memory_learning_summary = agent_result.memory_learning_summary;
    result.events = std::move(agent_result.events);
    result.trace = std::move(agent_result.trace);
    if (!result.ok && request.execution_control.should_stop()) {
        result.cancelled = true;
        result.failure_class = classify_execution_control_failure(request.execution_control);
        result.response_generation_status = common_agent_generation_status::cancelled;
        result.response_stop_reason = common_agent_generation_stop_reason::cancelled;
        error = request.execution_control.stop_reason();
    }
    compact_session_policy_pack_after_reflection(agent_result, result);
    result.error = ok ? std::string() : error;
    return ok;
}

bool common_agent_runtime_session_host::prepare_model(
        const common_agent_runtime_session_host_turn_request & request,
        std::string & error) {
    if (request.session_id.empty() || request.namespace_id.empty()) {
        error = "model preparation requires session and namespace identity";
        return false;
    }
    bool reused = false;
    if (!ensure_runtime(request, reused, error)) return false;
    return runtime->prepare_model(error);
}

const common_agent_runtime_session * common_agent_runtime_session_host::session() const {
    return runtime ? &runtime->runtime_host().session() : nullptr;
}

common_agent_runtime_session * common_agent_runtime_session_host::session() {
    return runtime ? &runtime->runtime_host().session() : nullptr;
}

common_agent_runtime_session_host_descriptor common_agent_runtime_session_host::describe_session() const {
    const auto resolved_policy_pack = active_policy_pack.has_value()
        ? active_policy_pack
        : config.resident_request.policy_pack;
    return {
        active_runtime_key.namespace_id,
        active_runtime_key.session_id,
        active_runtime_key.project_id,
        active_runtime_key.memory_scope,
        active_runtime_key.plan_scope,
        resolved_policy_pack.has_value() ? resolved_policy_pack->id : std::string(),
    };
}

void common_agent_runtime_session_host::reset() {
    runtime.reset();
    active_runtime_key = {};
    active_policy_pack.reset();
    generated_turn_counter = 0;
}
