#include "../agent-daemon-flydelta-internal.h"

#include "../../openapi/agent-openapi-flydelta-oracle.h"

#include <fstream>
extern "C" {
#include "hash/sha256/sha256.h"
}

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

bool daemon_flydelta_execution_identity(
        const daemon_options & options,
        const std::shared_ptr<common_agent_server_context_host> & host,
        common_flydelta_teaching_material_identity & identity, std::string & error) {
    identity = {};
    error.clear();
    const auto * ctx = host ? host->server().get_llama_context() : nullptr;
    const auto * model = ctx ? llama_get_model(ctx) : nullptr;
    if (!model) { error = "FlyDelta execution identity requires the resident model"; return false; }
    // Hash all GGUF shards. Tokenizer identity deliberately partitions by the
    // whole artifact: conservative, but never equates different vocabularies,
    // merge tables or tokenizer pre-processing metadata.
    char split_value[32] = {};
    int split_count = 1;
    if (llama_model_meta_val_str(model, "split.count", split_value, sizeof(split_value)) > 0) {
        try { split_count = std::stoi(split_value); }
        catch (...) { error = "FlyDelta model shard metadata is invalid"; return false; }
    }
    if (split_count < 1 || split_count > 10000) { error = "FlyDelta model shard count is invalid"; return false; }
    const auto & path = host->load_key().model;
    std::vector<char> prefix(path.size() + 64);
    if (split_count > 1 && llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), 0, split_count) <= 0) {
        error = "FlyDelta execution identity cannot resolve model shards"; return false;
    }
    json shards = json::array();
    for (int i = 0; i < split_count; ++i) {
        std::string shard = path;
        if (split_count > 1) {
            std::vector<char> name(path.size() + 64);
            if (llama_split_path(name.data(), name.size(), prefix.data(), i, split_count) <= 0) {
                error = "FlyDelta execution identity cannot resolve a model shard"; return false;
            }
            shard = name.data();
        }
        std::ifstream input(shard, std::ios::binary);
        if (!input) { error = "FlyDelta execution identity cannot read model artifact"; return false; }
        sha256_t state;
        sha256_init(&state);
        std::array<char, 65536> buffer;
        while (input) {
            input.read(buffer.data(), buffer.size());
            sha256_update(&state, reinterpret_cast<const unsigned char *>(buffer.data()), static_cast<size_t>(input.gcount()));
        }
        if (!input.eof()) { error = "FlyDelta model artifact fingerprint read failed"; return false; }
        unsigned char digest[SHA256_DIGEST_SIZE];
        sha256_final(&state, digest);
        static const char hex[] = "0123456789abcdef";
        std::string hash;
        for (const auto byte : digest) { hash += hex[byte >> 4]; hash += hex[byte & 15]; }
        shards.push_back(hash);
    }
    const auto fingerprint = [](const json & material) {
        const auto bytes = material.dump();
        return "sha256:" + hash_sha256_hex(bytes.data(), bytes.size());
    };
    identity.model_profile_fingerprint = fingerprint(json{{"revision", "resident-model-v1"},
        {"shards", shards}, {"profile", options.model_profile},
        {"host_label", options.adaptation_flydelta_model_profile_fingerprint}});
    identity.tokenizer_fingerprint = fingerprint(json{{"revision", "artifact-tokenizer-v1"}, {"shards", shards}});
    const auto meta = host->server().get_meta();
    const auto * templates = meta.chat_params.tmpls.get();
    if (!templates) { error = "FlyDelta resident chat templates are unavailable"; return false; }
    identity.template_fingerprint = fingerprint(json{{"revision", "resident-templates-v1"},
        {"default", common_chat_templates_source(templates)},
        {"tool_use", common_chat_templates_source(templates, "tool_use")}});
    identity.capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    identity.execution_context_fingerprint = fingerprint(json{{"revision", "server-context-capture-v1"},
        {"context_tokens", meta.slot_n_ctx}, {"capture_position", "generation_boundary"},
        {"model", identity.model_profile_fingerprint}, {"template", identity.template_fingerprint}});
    identity.scope_fingerprint = "daemon:teaching-material-v1";
    identity.verifier_revision = "daemon:host-verifier-v1";
    return common_flydelta_teaching_material_identity_validate(identity, error);
}

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
    common_flydelta_teaching_material_identity execution;
    if (!daemon_flydelta_execution_identity(options, host, execution, error)) return nullptr;
    if (teaching_material_runtime && !common_flydelta_teaching_material_execution_matches(
            teaching_material_runtime->identity(), execution.model_profile_fingerprint,
            execution.tokenizer_fingerprint, execution.template_fingerprint,
            execution.execution_context_fingerprint, execution.capture_layout_revision, error)) return nullptr;
    provider->model_profile_fingerprint = execution.model_profile_fingerprint;
    provider->capture_layout_revision = options.adaptation_flydelta_capture_layout_revision;
    provider->teaching_material_runtime = teaching_material_runtime;
    provider->n_predict = options.n_predict;
    provider->n_threads = options.n_threads;
    provider->model_n_embd = static_cast<size_t>(llama_model_n_embd(model));
    provider->model_n_layers = static_cast<size_t>(llama_model_n_layer(model));
    provider->dose_mode = options.adaptation_flydelta_dose_mode;
    provider->dose_calibration_ref = options.adaptation_flydelta_dose_calibration_ref;
    provider->search_model_call_budget = options.adaptation_flydelta_search_model_call_budget;
    provider->outcome_guidance_config = options.adaptation_flydelta_outcome_guidance;
    provider->outcome_projection_cache->set_budget(provider->outcome_guidance_config.projection_cache_bytes);
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
