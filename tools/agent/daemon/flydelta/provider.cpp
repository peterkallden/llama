#include "../agent-daemon-flydelta-internal.h"

#include "../../openapi/agent-openapi-flydelta-oracle.h"

#include <fstream>

namespace {

bool register_configured_openapi_oracles(
        const daemon_options & options,
        common_flydelta_oracle_registry & registry,
        std::string & error) {
    for (const auto & provider_config : options.openapi_providers) {
        if (!provider_config.enabled) continue;
        const std::filesystem::path configured_spec_path(provider_config.spec_path);
        const std::filesystem::path spec_path = configured_spec_path.is_absolute() ||
                provider_config.source_directory.empty()
            ? configured_spec_path
            : std::filesystem::path(provider_config.source_directory) / configured_spec_path;
        std::ifstream spec_file(spec_path);
        if (!spec_file) {
            if (provider_config.required) {
                error = "required OpenAPI spec could not be opened for FlyDelta Oracle: " +
                    spec_path.string();
                return false;
            }
            continue;
        }
        nlohmann::json spec;
        try {
            spec_file >> spec;
        } catch (const std::exception & exception) {
            if (provider_config.required) {
                error = "required OpenAPI spec could not be parsed for FlyDelta Oracle: " +
                    std::string(exception.what());
                return false;
            }
            continue;
        }
        agent_openapi_catalog catalog;
        std::string catalog_error;
        if (!build_agent_openapi_catalog(spec, provider_config, catalog, catalog_error)) {
            if (provider_config.required) {
                error = "required OpenAPI catalog could not be built for FlyDelta Oracle: " +
                    catalog_error;
                return false;
            }
            continue;
        }
        if (!register_agent_openapi_flydelta_oracles(
                catalog, registry, "openapi-v1", error)) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace agent_daemon_flydelta_internal {

std::shared_ptr<daemon_flydelta_resource_provider> daemon_flydelta_make_resource_provider(
        const daemon_options & options,
        const std::shared_ptr<common_agent_server_context_host> & host,
        agent_resource_store * resources,
        const std::shared_ptr<common_flydelta_teaching_material_runtime> & teaching_material_runtime,
        std::shared_ptr<common_learning_lifecycle_store> lifecycle_store,
        std::string & error) {
    error.clear();
    if (!host || resources == nullptr) {
        error = "FlyDelta resource provider requires a resident host and resource store";
        return nullptr;
    }
    const auto * context = host->server().get_llama_context();
    const auto * model = context == nullptr ? nullptr : llama_get_model(context);
    if (model == nullptr || llama_model_n_embd(model) <= 0 || llama_model_n_layer(model) <= 2) {
        error = "FlyDelta resource provider requires a loaded compatible model";
        return nullptr;
    }
    auto provider = std::make_shared<daemon_flydelta_resource_provider>();
    provider->host = host;
    provider->resources = resources;
    // The provider starts with the portable resource-contract defaults.
    // Worker callbacks replace this view with job.seed.scope before
    // resolving job-owned resources.
    provider->authority = {};
    provider->model_profile_fingerprint = options.adaptation_flydelta_model_profile_fingerprint.empty()
        ? "model:" + std::filesystem::path(options.model).filename().string()
        : options.adaptation_flydelta_model_profile_fingerprint;
    provider->capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    provider->teaching_material_runtime = teaching_material_runtime;
    provider->n_predict = options.n_predict;
    provider->n_threads = options.n_threads;
    provider->model_n_embd = static_cast<size_t>(llama_model_n_embd(model));
    provider->model_n_layers = static_cast<size_t>(llama_model_n_layer(model));
    provider->oracle_registry = common_flydelta_make_default_oracle_registry();
    if (!register_configured_openapi_oracles(options, provider->oracle_registry, error)) {
        return nullptr;
    }
    provider->lifecycle_store = std::move(lifecycle_store);
    if (!provider->lifecycle_store) {
        auto lifecycle = make_agent_learning_lifecycle_store(
            options.adaptation_flydelta_lifecycle_backend,
            options.adaptation_flydelta_lifecycle_path, error);
        if (!lifecycle) return nullptr;
        provider->lifecycle_store = std::shared_ptr<common_learning_lifecycle_store>(
            std::move(lifecycle));
    }
    return provider;
}

} // namespace agent_daemon_flydelta_internal
