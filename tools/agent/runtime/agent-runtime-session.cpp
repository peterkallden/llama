#include "agent-runtime-session.h"

#include "../cli/agent-cli-inference.h"
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
#include "../runtime/agent-server-context-host.h"
#include "agent-model-loaders.h"
#endif

#include "chat.h"

#include <algorithm>
#include <filesystem>
#include <utility>

namespace {

common_agent_model_load_key make_agent_model_load_key(
        const common_agent_inference_options & options) {
    return {
        options.model,
        options.n_gpu_layers,
        options.fit_params,
        options.mmproj,
    };
}

common_agent_inference_context_key make_agent_inference_context_key(
        const common_agent_inference_options & options,
        agent_inference_backend backend) {
    return {
        backend,
        make_agent_model_load_key(options),
    };
}

bool common_agent_model_load_key_match(
        const common_agent_model_load_key & lhs,
        const common_agent_model_load_key & rhs) {
    return lhs.model == rhs.model &&
        lhs.mmproj == rhs.mmproj &&
        lhs.n_gpu_layers == rhs.n_gpu_layers &&
        lhs.fit_params == rhs.fit_params;
}

bool common_agent_inference_context_key_match(
        const common_agent_inference_context_key & lhs,
        const common_agent_inference_context_key & rhs) {
    return lhs.backend == rhs.backend &&
        common_agent_model_load_key_match(lhs.model_key, rhs.model_key);
}

bool build_agent_inference_session(
        const common_agent_inference_options & options,
        agent_inference_backend backend,
        llama_model * model,
        const common_chat_templates * templates,
        std::shared_ptr<common_agent_server_context_host> server_context_host,
        const std::vector<llama_adapter_lora *> & adapters,
        const std::vector<float> & adapter_scales,
        common_agent_inference_session & session,
        std::string & error) {
    session = {};
    session.backend = backend;
    session.model = model;
    session.templates = templates;
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
    if (backend == agent_inference_backend::server_context) {
        if (!adapters.empty()) {
            error = "server-context inference backend does not support runtime adapter overlays";
            return false;
        }
        if (!server_context_host) {
            error = "server_context host is not loaded";
            return false;
        }
        if (!server_context_host->build_inference_session(session, error)) {
            return false;
        }
        session.keepalive = std::move(server_context_host);
        error.clear();
        return true;
    }
#else
    if (backend == agent_inference_backend::server_context) {
        error = "server-context inference backend is unavailable in this host";
        return false;
    }
#endif

    session.inference = make_llama_cli_agent_inference(
        model, templates, adapters, adapter_scales);
    error.clear();
    return true;
}

} // namespace

void common_agent_runtime_loaded_model_state::reset() {
    for (auto * adapter : adapters) {
        llama_adapter_lora_free(adapter);
    }
    adapters.clear();
    adapter_scales.clear();
    chat_templates.reset();
    server_context_host.reset();
    if (model != nullptr && !externally_owned) {
        llama_model_free(model);
    }
    model = nullptr;
    loaded = false;
    backend = agent_inference_backend::cli;
    key = {};
    profile_id.clear();
    profile_cache_key.clear();
    chat_templates_view = nullptr;
    externally_owned = false;
    residency_owner.reset();
}

void common_agent_runtime_inference_context_state::reset() {
    session = {};
    initialized = false;
    key = {};
}

common_agent_runtime_session & common_agent_runtime_session::operator=(common_agent_runtime_session && other) {
    if (this != &other) {
        reset();
        loaded_model = std::move(other.loaded_model);
        inference_context = std::move(other.inference_context);
        other.loaded_model.loaded = false;
        other.loaded_model.model = nullptr;
        other.loaded_model.server_context_host.reset();
        other.loaded_model.adapters.clear();
        other.loaded_model.adapter_scales.clear();
        other.loaded_model.profile_id.clear();
        other.loaded_model.profile_cache_key.clear();
        other.loaded_model.chat_templates_view = nullptr;
        other.loaded_model.externally_owned = false;
        other.loaded_model.residency_owner.reset();
        other.inference_context.initialized = false;
    }
    return *this;
}

common_agent_runtime_session::~common_agent_runtime_session() {
    reset();
}

const common_agent_inference_session * common_agent_runtime_session::active_inference_session() const {
    return inference_context.initialized ? &inference_context.session : nullptr;
}

common_agent_inference_session * common_agent_runtime_session::active_inference_session() {
    return inference_context.initialized ? &inference_context.session : nullptr;
}

void common_agent_runtime_session::reset() {
    inference_context.reset();
    loaded_model.reset();
}

bool initialize_agent_runtime_session(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    bool memory_enabled,
    const std::string & fallback_reason,
    common_agent_runtime_session & session,
    std::string & error) {
    (void) memory_enabled;
    (void) fallback_reason;

    const auto requested_model_key = make_agent_model_load_key(options);
    const auto requested_context_key = make_agent_inference_context_key(options, backend);

    if (session.inference_context.initialized &&
            session.inference_context.session.inference &&
            common_agent_inference_context_key_match(session.inference_context.key, requested_context_key)) {
        error.clear();
        return true;
    }

    const bool reuse_loaded_model =
        session.loaded_model.loaded &&
        session.loaded_model.backend == backend &&
        common_agent_model_load_key_match(session.loaded_model.key, requested_model_key);

    session.inference_context.reset();

    if (!reuse_loaded_model) {
        session.loaded_model.reset();
    }

    if (backend == agent_inference_backend::cli) {
        if (!options.mmproj.empty()) {
            error = "CLI inference backend does not support mmproj yet; use server-context";
            return false;
        }
        if (!session.loaded_model.loaded) {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = options.n_gpu_layers;
            session.loaded_model.model = llama_model_load_from_file(options.model.c_str(), model_params);
            if (session.loaded_model.model == nullptr) {
                error = "failed to load model: " + options.model;
                return false;
            }
            session.loaded_model.chat_templates = common_chat_templates_init(session.loaded_model.model, "");
            session.loaded_model.chat_templates_view = session.loaded_model.chat_templates.get();
            session.loaded_model.loaded = true;
            session.loaded_model.backend = backend;
            session.loaded_model.key = requested_model_key;
        }
#ifndef LLAMA_AGENT_ANDROID_CLI_ONLY
    } else if (backend == agent_inference_backend::server_context) {
        if (!session.loaded_model.loaded) {
            auto host = std::make_shared<common_agent_server_context_host>();
            if (!host->start(make_agent_server_context_host_config(options), error)) {
                return false;
            }
            session.loaded_model.server_context_host = std::move(host);
            session.loaded_model.loaded = true;
            session.loaded_model.backend = backend;
            session.loaded_model.key = requested_model_key;
        }
    } else {
        session.loaded_model.reset();
    }
#else
    }
#endif

    if (!build_agent_inference_session(
            options,
            backend,
            session.loaded_model.model,
            session.loaded_model.chat_templates_view,
            session.loaded_model.server_context_host,
            session.loaded_model.adapters,
            session.loaded_model.adapter_scales,
            session.inference_context.session,
            error)) {
        session.reset();
        return false;
    }

    session.inference_context.initialized = true;
    session.inference_context.key = requested_context_key;
    session.inference_context.session.profile_id = session.loaded_model.profile_id;
    session.inference_context.session.profile_cache_key = session.loaded_model.profile_cache_key;
    error.clear();
    return true;
}

bool initialize_agent_runtime_session_from_resident_model(
    const common_agent_inference_options & options,
    agent_inference_backend backend,
    const std::shared_ptr<common_agent_runtime_resident_model> & resident_model,
    common_agent_runtime_session & session,
    std::string & error) {
#ifdef LLAMA_AGENT_ANDROID_CLI_ONLY
    (void) options;
    (void) backend;
    (void) resident_model;
    session.reset();
    error = "resident model loading is unavailable in the Android CLI-only runtime";
    return false;
#else
    error.clear();
    const auto loaded = common_agent_runtime_loaded_model_cast(resident_model);
    if (!loaded) {
        error = "residency loader returned an unknown model resource";
        return false;
    }
    const std::string expected_backend = backend == agent_inference_backend::cli
        ? "cli" : "server-context";
    if (loaded->selection.backend != expected_backend) {
        error = "resident model backend does not match the requested inference backend";
        return false;
    }
    if (!loaded->selection.adapters.empty()) {
        error = "resident model adapter overlays are not supported by the runtime loaders yet";
        return false;
    }
    if (backend == agent_inference_backend::cli && loaded->model == nullptr) {
        error = "resident CLI model resource has no llama model";
        return false;
    }
    if (backend == agent_inference_backend::server_context &&
            !loaded->server_context_host) {
        error = "resident server-context model resource has no server context host";
        return false;
    }

    const auto requested_context_key = make_agent_inference_context_key(options, backend);
    if (session.inference_context.initialized &&
            session.inference_context.session.inference &&
            common_agent_inference_context_key_match(
                session.inference_context.key, requested_context_key)) {
        error.clear();
        return true;
    }

    session.reset();
    session.loaded_model.model = loaded->model;
    session.loaded_model.chat_templates_view = loaded->chat_templates.get();
    session.loaded_model.server_context_host = loaded->server_context_host;
    session.loaded_model.loaded = true;
    session.loaded_model.backend = backend;
    session.loaded_model.key = make_agent_model_load_key(options);
    session.loaded_model.profile_id = loaded->selection.profile_id;
    session.loaded_model.profile_cache_key =
        common_agent_model_selection_cache_key(loaded->selection);
    session.loaded_model.externally_owned = true;
    session.loaded_model.residency_owner = resident_model;

    if (!build_agent_inference_session(
            options,
            backend,
            session.loaded_model.model,
            session.loaded_model.chat_templates_view,
            session.loaded_model.server_context_host,
            session.loaded_model.adapters,
            session.loaded_model.adapter_scales,
            session.inference_context.session,
            error)) {
        session.reset();
        return false;
    }
    session.inference_context.initialized = true;
    session.inference_context.key = requested_context_key;
    session.inference_context.session.profile_id = session.loaded_model.profile_id;
    session.inference_context.session.profile_cache_key = session.loaded_model.profile_cache_key;
    return true;
#endif
}

bool apply_agent_runtime_model_profile(
        common_agent_runtime_session & session,
        const common_agent_model_profile & profile,
        const common_learning_adapter_registry & registry,
        const std::string & adapter_root,
        std::string & error) {
    error.clear();
    if (!session.loaded_model.loaded || session.loaded_model.model == nullptr) {
        error = "cannot apply a model profile before the base model is loaded";
        return false;
    }
    if (!common_agent_validate_model_profile(profile, error)) {
        return false;
    }
    if (profile.adapters.empty() && !adapter_root.empty()) {
        // An empty overlay is the explicit baseline.  Do not require an
        // artifact directory merely to select the adapter-free profile.
    } else if (!profile.adapters.empty() && adapter_root.empty()) {
        error = "adapter root is required for a profile with overlays";
        return false;
    }

    std::vector<common_learning_adapter_manifest> manifests;
    if (!registry.resolve_active_overlays(profile, manifests, error)) {
        return false;
    }

    if (session.loaded_model.profile_cache_key ==
            common_agent_model_profile_cache_key(profile) &&
            session.inference_context.initialized) {
        return true;
    }

    std::vector<llama_adapter_lora *> loaded_adapters;
    std::vector<float> scales;
    const std::filesystem::path root(adapter_root);
    for (const auto & requested : profile.adapters) {
        const auto manifest = std::find_if(manifests.begin(), manifests.end(),
            [&](const auto & item) { return item.id == requested.adapter_id; });
        if (manifest == manifests.end()) {
            error = "resolved adapter is missing from profile: " + requested.adapter_id;
            for (auto * adapter : loaded_adapters) llama_adapter_lora_free(adapter);
            return false;
        }
        const auto path = root / manifest->artifact_path;
        if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
            error = "adapter artifact is not a regular file: " + path.string();
            for (auto * adapter : loaded_adapters) llama_adapter_lora_free(adapter);
            return false;
        }
        auto * adapter = llama_adapter_lora_init(
            session.loaded_model.model, path.string().c_str());
        if (adapter == nullptr) {
            error = "failed to load adapter artifact: " + path.string();
            for (auto * loaded : loaded_adapters) llama_adapter_lora_free(loaded);
            return false;
        }
        loaded_adapters.push_back(adapter);
        scales.push_back(static_cast<float>(requested.scale));
    }

    session.inference_context.reset();
    for (auto * adapter : session.loaded_model.adapters) {
        llama_adapter_lora_free(adapter);
    }
    session.loaded_model.adapters = std::move(loaded_adapters);
    session.loaded_model.adapter_scales = std::move(scales);
    session.loaded_model.profile_id = profile.id;
    session.loaded_model.profile_cache_key = common_agent_model_profile_cache_key(profile);
    common_agent_inference_options options;
    options.model = session.loaded_model.key.model;
    options.n_gpu_layers = session.loaded_model.key.n_gpu_layers;
    options.fit_params = session.loaded_model.key.fit_params;
    options.mmproj = session.loaded_model.key.mmproj;
    if (!build_agent_inference_session(
            options,
            session.loaded_model.backend,
            session.loaded_model.model,
            session.loaded_model.chat_templates_view,
            session.loaded_model.server_context_host,
            session.loaded_model.adapters,
            session.loaded_model.adapter_scales,
            session.inference_context.session,
            error)) {
        session.loaded_model.reset();
        return false;
    }
    session.inference_context.initialized = true;
    session.inference_context.key = {
        session.loaded_model.backend,
        session.loaded_model.key,
    };
    session.inference_context.session.profile_id = session.loaded_model.profile_id;
    session.inference_context.session.profile_cache_key = session.loaded_model.profile_cache_key;
    return true;
}
