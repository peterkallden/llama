#include "agent-runtime-resident.h"

#include "../runtime/agent-runtime-host.h"

bool common_agent_runtime_resident_host::run_turn(
        common_agent_runtime_host_inputs & inputs,
        common_agent_result & result,
        std::string & error) {
    const bool original_reset = inputs.reset_session_on_completion;
    inputs.reset_session_on_completion = false;
    const bool ok = run_agent_runtime_host_turn(inputs, runtime_session, result, error);
    inputs.reset_session_on_completion = original_reset;
    return ok;
}

void common_agent_runtime_resident_host::reset() {
    runtime_session.reset();
}

common_agent_runtime_turn_request make_agent_runtime_resident_base_turn_request(
        const common_agent_runtime_resident_request_config & config) {
    common_agent_runtime_turn_request turn_request;
    turn_request.request.prompt = config.prompt;
    turn_request.request.messages = {{"user", config.prompt}};
    turn_request.request.session_id = config.session_id;
    turn_request.request.namespace_id = config.namespace_id;
    turn_request.request.project_id = config.project_id;
    turn_request.request.policy_pack = config.policy_pack;
    turn_request.scope.namespace_id = config.namespace_id;
    turn_request.scope.session_id = config.session_id;
    turn_request.scope.project_id = config.project_id;
    turn_request.scope.memory_scope = config.memory_scope;
    turn_request.scope.plan_scope = config.plan_scope;
    turn_request.inference_options.model = config.model;
    turn_request.inference_options.mmproj = config.mmproj;
    turn_request.inference_options.n_predict = config.n_predict;
    turn_request.inference_options.n_gpu_layers = config.n_gpu_layers;
    turn_request.inference_options.fit_params = config.fit_params;
    turn_request.inference_options.n_threads = config.n_threads;
    turn_request.inference_options.context_size_tokens = config.context_size_tokens;
    turn_request.policy.agent_inference_backend = config.inference_backend;
    turn_request.orchestration_config.prompt = config.prompt;
    turn_request.generation_options.n_predict = config.n_predict;
    turn_request.generation_options.n_threads = config.n_threads;
    return turn_request;
}

common_agent_runtime_turn_request make_agent_runtime_resident_turn_request(
        const common_agent_runtime_turn_request & base_turn_request,
        const std::string & prompt,
        const std::string & turn_id,
        int n_predict_override) {
    auto turn_request = base_turn_request;
    turn_request.request.prompt = prompt;
    turn_request.request.messages = {{"user", prompt}};
    turn_request.request.turn_id = turn_id;
    turn_request.scope.turn_id = turn_id;
    turn_request.orchestration_config.prompt = prompt;
    if (n_predict_override > 0) {
        turn_request.inference_options.n_predict = n_predict_override;
        turn_request.generation_options.n_predict = n_predict_override;
    }
    return turn_request;
}

common_agent_runtime_resident_runtime_config make_agent_runtime_resident_runtime_config(
        common_memory_store & memory_store,
        common_plan_store * plan_store,
        common_agent_runtime_turn_request base_turn_request,
        std::string current_plan_id,
        std::vector<common_blueprint_candidate> installed_blueprint_candidates,
        common_agent_runtime_tooling tooling,
        std::shared_ptr<common_agent_runtime_model_residency> model_residency,
        std::string model_profile_id) {
    return {
        memory_store,
        plan_store,
        std::move(base_turn_request),
        std::move(current_plan_id),
        std::move(installed_blueprint_candidates),
        std::move(tooling),
        std::move(model_residency),
        std::move(model_profile_id),
    };
}

common_agent_runtime_resident_runtime::common_agent_runtime_resident_runtime(
        common_agent_runtime_resident_runtime_config config)
    : memory_store(config.memory_store),
      plan_store(config.plan_store),
      base_turn_request(std::move(config.base_turn_request)),
      resident_current_plan_id(std::move(config.current_plan_id)),
      installed_blueprint_candidates(std::move(config.installed_blueprint_candidates)),
      tooling(std::move(config.tooling)),
      model_residency(std::move(config.model_residency)),
      model_profile_id(std::move(config.model_profile_id)) {}

void common_agent_runtime_resident_runtime::set_tooling(common_agent_runtime_tooling next_tooling) {
    tooling = std::move(next_tooling);
}

void common_agent_runtime_resident_runtime::set_execution_control(
        common_agent_runtime_execution_control execution_control) {
    base_turn_request.execution_control = std::move(execution_control);
}

void common_agent_runtime_resident_runtime::set_policy_pack(
        std::optional<common_memory_policy_pack> policy_pack) {
    base_turn_request.request.policy_pack = std::move(policy_pack);
}

bool common_agent_runtime_resident_runtime::prepare_model(std::string & error) {
    if (model_residency) {
        if (resident_model_handle.valid()) {
            error.clear();
            return true;
        }
        if (!model_residency->acquire(model_profile_id, resident_model_handle, error)) {
            return false;
        }
        base_turn_request.inference_options.model = resident_model_handle.selection.path;
        base_turn_request.inference_options.mmproj = resident_model_handle.selection.mmproj;
        base_turn_request.inference_options.context_size_tokens =
            resident_model_handle.selection.context_size_tokens;
        base_turn_request.policy.agent_inference_backend = resident_model_handle.selection.backend;
        agent_inference_backend backend;
        if (!parse_agent_inference_backend(
                base_turn_request.policy.agent_inference_backend, backend) ||
                !initialize_agent_runtime_session_from_resident_model(
                    base_turn_request.inference_options,
                    backend,
                    resident_model_handle.model,
                    host.session(),
                    error)) {
            std::string release_error;
            model_residency->release(resident_model_handle, release_error);
            resident_model_handle = {};
            return false;
        }
        return true;
    }
    agent_inference_backend backend = agent_inference_backend::cli;
    if (!parse_agent_inference_backend(
            base_turn_request.policy.agent_inference_backend, backend)) {
        error = "unsupported inference backend: " +
            base_turn_request.policy.agent_inference_backend;
        return false;
    }
    return initialize_agent_runtime_session(
        base_turn_request.inference_options,
        backend,
        base_turn_request.memory_enabled,
        base_turn_request.fallback_reason,
        host.session(),
        error);
}

bool common_agent_runtime_resident_runtime::run_chat_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        int n_predict,
        common_agent_result & result,
        std::string & error) {
    if (!prepare_model(error)) return false;
    const std::vector<common_memory_hit> memories;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        nullptr,
        make_agent_runtime_resident_turn_request(base_turn_request, prompt, turn_id, n_predict),
        nullptr,
        nullptr,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_chat_inputs(build_context);
    return host.run_turn(inputs, result, error);
}

bool common_agent_runtime_resident_runtime::run_agent_prompt(
        const std::string & prompt,
        const std::string & turn_id,
        int n_predict,
        common_agent_result & result,
        std::string & error) {
    if (!prepare_model(error)) return false;
    if (plan_store == nullptr) {
        error = "resident agent runtime requires a plan store";
        return false;
    }

    const std::vector<common_memory_hit> memories;
    common_agent_runtime_host_build_context build_context{
        memory_store,
        plan_store,
        make_agent_runtime_resident_turn_request(base_turn_request, prompt, turn_id, n_predict),
        &resident_current_plan_id,
        &installed_blueprint_candidates,
        memories,
        tooling,
    };
    auto inputs = make_agent_runtime_host_agent_inputs(build_context, build_context.turn_request.orchestration_config);
    const bool ok = host.run_turn(inputs, result, error);
    if (ok && result.plan_id) {
        resident_current_plan_id = *result.plan_id;
    }
    return ok;
}

void common_agent_runtime_resident_runtime::reset() {
    host.reset();
    if (model_residency && resident_model_handle.valid()) {
        std::string ignored;
        model_residency->release(resident_model_handle, ignored);
    }
    resident_model_handle = {};
    resident_current_plan_id.clear();
}
