#include "agent/adaptation/flydelta/flydelta-model-adapter.h"
#include "agent/adaptation/flydelta/flydelta-evaluator.h"

bool common_flydelta_model_host_validate(
        const common_flydelta_model_host & host,
        std::string & error) {
    error.clear();
    if (!host.register_evaluator) {
        error = "FlyDelta model host has no evaluator registration callback";
        return false;
    }
    if (!host.capabilities.bootstrap_zoom &&
            !host.capabilities.adaptive_alpha &&
            !host.capabilities.teacher_forced_margin &&
            !host.capabilities.orthogonal_search &&
            !host.capabilities.representation_augmentation &&
            !host.capabilities.host_verification) {
        error = "FlyDelta model host advertises no capabilities";
        return false;
    }
    return true;
}

bool common_flydelta_model_adapter_validate(
        const common_flydelta_model_adapter & adapter,
        std::string & error) {
    error.clear();
    if (!adapter.worker_callback) {
        error = "FlyDelta model adapter has no worker callback";
        return false;
    }
    if (!adapter.capabilities.bootstrap_zoom &&
            !adapter.capabilities.adaptive_alpha &&
            !adapter.capabilities.teacher_forced_margin &&
            !adapter.capabilities.orthogonal_search &&
            !adapter.capabilities.representation_augmentation &&
            !adapter.capabilities.host_verification) {
        error = "FlyDelta model adapter advertises no capabilities";
        return false;
    }
    return true;
}

bool common_flydelta_model_adapter_supports_search(
        const common_flydelta_model_adapter & adapter) {
    return static_cast<bool>(adapter.worker_callback) &&
        (adapter.capabilities.bootstrap_zoom ||
         adapter.capabilities.adaptive_alpha ||
         adapter.capabilities.orthogonal_search ||
         adapter.capabilities.representation_augmentation);
}

std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_evaluator(
        const common_flydelta_evaluator_config & config,
        const common_flydelta_evaluator_callbacks & callbacks,
        common_flydelta_model_capabilities capabilities,
        std::string & error) {
    auto adapter = std::make_shared<common_flydelta_model_adapter>();
    adapter->capabilities = capabilities;
    adapter->worker_callback = [config, callbacks](
            const common_flydelta_experiment_job & job,
            common_flydelta_experiment_worker_result & result,
            std::string & callback_error) {
        common_flydelta_evaluator_result evaluated;
        if (!common_flydelta_evaluate_job(
                job, config, callbacks, evaluated, callback_error)) {
            return false;
        }
        return common_flydelta_worker_result_from_evaluator(
            evaluated, result, callback_error);
    };
    if (!common_flydelta_model_adapter_validate(*adapter, error)) {
        return {};
    }
    error.clear();
    return adapter;
}

std::shared_ptr<const common_flydelta_model_adapter>
common_flydelta_model_adapter_from_host(
        const common_flydelta_model_host & host,
        std::string & error) {
    if (!common_flydelta_model_host_validate(host, error)) {
        return {};
    }

    common_flydelta_evaluator_config config;
    common_flydelta_evaluator_callbacks callbacks;
    if (!host.register_evaluator(config, callbacks, error)) {
        if (error.empty()) {
            error = "FlyDelta model host failed to register evaluator";
        }
        return {};
    }
    if (host.capabilities.orthogonal_search &&
            !callbacks.run_search_pipeline_with_search_state) {
        error = "FlyDelta orthogonal capability requires a post-Bootstrap state-aware runner";
        return {};
    }
    return common_flydelta_model_adapter_from_evaluator(
        config, callbacks, host.capabilities, error);
}
